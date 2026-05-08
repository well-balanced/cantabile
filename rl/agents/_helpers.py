from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from networks import MLP, Ensemble, StateActionValue

LogDict = dict[str, float]


@dataclass(frozen=True)
class SACConfig:
    num_qs: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    hidden_dims: Sequence[int] = (256, 256, 256)
    activation: str = "gelu"
    num_min_qs: Optional[int] = None
    critic_dropout_rate: float = 0.0
    critic_layer_norm: bool = False
    tau: float = 0.005
    target_entropy: Optional[float] = None
    init_temperature: float = 1.0
    backup_entropy: bool = True
    backup_entropy_event: bool = True


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda _: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


@partial(jax.jit, static_argnames="apply_fn")
def sample_actions(rng, apply_fn, params, observations: np.ndarray) -> tuple[jnp.ndarray, Any]:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def eval_actions(apply_fn, params, observations: np.ndarray) -> jnp.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


def split_entropy_backup(
    temperature: jnp.ndarray,
    log_probs: jnp.ndarray,
    discount: jnp.ndarray,
    share: float,
) -> jnp.ndarray:
    return share * discount * temperature * log_probs


def dense_critic_obs(observations):
    """Observation routing for dense critic. Pass-through for flat arrays."""
    if isinstance(observations, dict):
        return observations.get("critic_dense", observations.get("critic", observations.get("actor")))
    return observations


def event_critic_obs(observations):
    """Observation routing for event critic. Pass-through for flat arrays."""
    if isinstance(observations, dict):
        return observations.get("critic_event", observations.get("critic_dense", observations.get("critic", observations.get("actor"))))
    return observations


def make_critic(config: SACConfig):
    critic_base_cls = partial(
        MLP,
        hidden_dims=config.hidden_dims,
        activation=getattr(nn, config.activation),
        activate_final=True,
        dropout_rate=config.critic_dropout_rate,
        use_layer_norm=config.critic_layer_norm,
    )
    critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
    return Ensemble(critic_cls, num=config.num_qs), Ensemble(critic_cls, num=config.num_min_qs or config.num_qs)


def make_temperature(config: SACConfig, key):
    temp_def = Temperature(config.init_temperature)
    temp_params = temp_def.init(key)["params"]
    return TrainState.create(
        apply_fn=temp_def.apply,
        params=temp_params,
        tx=optax.adam(learning_rate=config.temp_lr),
    )
