from dataclasses import dataclass

from functools import partial
from typing import Any, Dict, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState

from distributions import TanhNormal
from networks import MLP, Ensemble, StateActionValue, subsample_ensemble
from specs import EnvironmentSpec, zeros_like
from replay import Transition

LogDict = Dict[str, float]


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
def _sample_actions(
    rng, apply_fn, params, observations: np.ndarray
) -> tuple[jnp.ndarray, Any]:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> jnp.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


@dataclass(frozen=True)
class SACConfig:
    """Configuration options for SAC."""

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


class SAC(struct.PyTreeNode):
    """Soft-Actor Critic (SAC)."""

    actor: TrainState
    rng: Any
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)

    @staticmethod
    def initialize(
        spec: EnvironmentSpec,
        config: SACConfig,
        seed: int = 0,
        discount: float = 0.99,
    ) -> "SAC":
        """Initializes the agent from the given environment spec and config."""

        action_dim = spec.action.shape[-1]
        observations = zeros_like(spec.observation)
        actions = zeros_like(spec.action)

        target_entropy = config.target_entropy or -0.5 * action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activation=getattr(nn, config.activation),
            activate_final=True,
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=config.actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activation=getattr(nn, config.activation),
            activate_final=True,
            dropout_rate=config.critic_dropout_rate,
            use_layer_norm=config.critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config.num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=config.critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=config.num_min_qs or config.num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(config.init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=config.temp_lr),
        )

        return SAC(
            actor=actor,
            rng=rng,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=config.tau,
            discount=discount,
            num_qs=config.num_qs,
            num_min_qs=config.num_min_qs,
            backup_entropy=config.backup_entropy,
        )

    def update_actor(self, transitions: Transition) -> tuple["SAC", LogDict]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, transitions.state)
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                transitions.state,
                actions,
                True,  # training.
                rngs={"dropout": key2},
            )
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> tuple["SAC", LogDict]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, transitions: Transition) -> tuple["SAC", LogDict]:
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, transitions.next_state
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key=key,
            params=self.target_critic.params,
            num_sample=self.num_min_qs,
            num_qs=self.num_qs,
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            transitions.next_state,
            next_actions,
            True,  # training.
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)

        target_q = transitions.reward + self.discount * transitions.discount * next_q

        if self.backup_entropy:
            target_q -= (
                self.discount
                * transitions.discount
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                transitions.state,
                transitions.action,
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @jax.jit
    def update(self, transitions: Transition) -> tuple["SAC", LogDict]:
        new_agent = self
        new_agent, critic_info = new_agent.update_critic(transitions)
        new_agent, actor_info = new_agent.update_actor(transitions)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        return new_agent, {**actor_info, **critic_info, **temp_info}

    def sample_actions(self, observations: np.ndarray) -> tuple["SAC", np.ndarray]:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return self.replace(rng=new_rng), np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)

    def save(self, path: "str | Path") -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(serialization.to_bytes(self))

    @staticmethod
    def load(path: "str | Path", template: "SAC") -> "SAC":
        from pathlib import Path
        with Path(path).open("rb") as f:
            return serialization.from_bytes(template, f.read())


# ---------------------------------------------------------------------------
# Residual RL helpers
# ---------------------------------------------------------------------------

def _compose_actions(
    base_actions: jnp.ndarray,
    residual_actions: jnp.ndarray,
    alpha: float,
    action_min: jnp.ndarray,
    action_max: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.clip(base_actions + alpha * residual_actions, action_min, action_max)


def _expand_residual_actions(
    residual_actions: jnp.ndarray,
    action_indices: tuple,
    action_dim: int,
) -> jnp.ndarray:
    full = jnp.zeros(
        residual_actions.shape[:-1] + (action_dim,), dtype=residual_actions.dtype
    )
    return full.at[..., jnp.asarray(action_indices)].set(residual_actions)


class ResidualSAC(struct.PyTreeNode):
    """SAC with a frozen base actor and a trainable residual actor.

    final_action = clip(base_action + alpha * residual_action, action_min, action_max)

    The residual actor receives concat(obs, base_actions) as input and outputs
    actions only for the indices specified by `residual_action_indices`.
    The base actor is frozen (no gradient flows through it).
    """

    actor: TrainState
    base_actor: TrainState
    rng: Any
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    action_min: jnp.ndarray
    action_max: jnp.ndarray
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    residual_alpha: float = struct.field(pytree_node=False)
    residual_action_indices: tuple = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)

    @staticmethod
    def initialize(
        spec: EnvironmentSpec,
        config: SACConfig,
        base_actor: TrainState,
        seed: int = 0,
        discount: float = 0.99,
        residual_alpha: float = 0.1,
        residual_action_indices: Optional[Sequence[int]] = None,
    ) -> "ResidualSAC":
        full_action_dim = spec.action.shape[-1]
        observations = zeros_like(spec.observation)
        actions = zeros_like(spec.action)

        if residual_action_indices is None:
            residual_action_indices = tuple(range(full_action_dim))
        else:
            residual_action_indices = tuple(residual_action_indices)

        residual_action_dim = len(residual_action_indices)
        target_entropy = config.target_entropy or -0.5 * residual_action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        # Residual actor input: concat(obs, base_actions).
        base_actions = _eval_actions(base_actor.apply_fn, base_actor.params, observations)
        actor_inputs = jnp.concatenate([observations, base_actions], axis=-1)

        actor_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activation=getattr(nn, config.activation),
            activate_final=True,
        )
        actor_def = TanhNormal(actor_base_cls, residual_action_dim)
        actor_params = actor_def.init(actor_key, actor_inputs)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=config.actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activation=getattr(nn, config.activation),
            activate_final=True,
            dropout_rate=config.critic_dropout_rate,
            use_layer_norm=config.critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config.num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=config.critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=config.num_min_qs or config.num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(config.init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=config.temp_lr),
        )

        return ResidualSAC(
            actor=actor,
            base_actor=base_actor,
            rng=rng,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            action_min=jnp.asarray(spec.action.minimum),
            action_max=jnp.asarray(spec.action.maximum),
            tau=config.tau,
            discount=discount,
            target_entropy=target_entropy,
            num_qs=config.num_qs,
            num_min_qs=config.num_min_qs,
            backup_entropy=config.backup_entropy,
            residual_alpha=residual_alpha,
            residual_action_indices=residual_action_indices,
            action_dim=full_action_dim,
        )

    def _base_actions(self, observations: jnp.ndarray) -> jnp.ndarray:
        return _eval_actions(self.base_actor.apply_fn, self.base_actor.params, observations)

    def _actor_inputs(self, observations: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([observations, self._base_actions(observations)], axis=-1)

    def _composed_actions(
        self,
        observations: jnp.ndarray,
        residual_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        base = self._base_actions(observations)
        full_residual = _expand_residual_actions(
            residual_actions, self.residual_action_indices, self.action_dim
        )
        return _compose_actions(base, full_residual, self.residual_alpha, self.action_min, self.action_max)

    def update_actor(self, transitions: Transition) -> tuple["ResidualSAC", LogDict]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            actor_inputs = self._actor_inputs(transitions.state)
            dist = self.actor.apply_fn({"params": actor_params}, actor_inputs)
            residual_actions, log_probs = dist.sample_and_log_prob(seed=key)
            actions = self._composed_actions(transitions.state, residual_actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                transitions.state, actions, True,
                rngs={"dropout": key2},
            )
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> tuple["ResidualSAC", LogDict]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temperature_loss": temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info

    def update_critic(self, transitions: Transition) -> tuple["ResidualSAC", LogDict]:
        actor_inputs = self._actor_inputs(transitions.next_state)
        dist = self.actor.apply_fn({"params": self.actor.params}, actor_inputs)

        rng = self.rng
        key, rng = jax.random.split(rng)
        residual_next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
        next_actions = self._composed_actions(transitions.next_state, residual_next_actions)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key=key, params=self.target_critic.params,
            num_sample=self.num_min_qs, num_qs=self.num_qs,
        )
        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            transitions.next_state, next_actions, True,
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)
        target_q = transitions.reward + self.discount * transitions.discount * next_q
        if self.backup_entropy:
            target_q -= (
                self.discount * transitions.discount
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                transitions.state, transitions.action, True,
                rngs={"dropout": key},
            )
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @jax.jit
    def update(self, transitions: Transition) -> tuple["ResidualSAC", LogDict]:
        new_agent = self
        new_agent, critic_info = new_agent.update_critic(transitions)
        new_agent, actor_info = new_agent.update_actor(transitions)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        return new_agent, {**actor_info, **critic_info, **temp_info}

    def sample_actions(self, observations: np.ndarray) -> tuple["ResidualSAC", np.ndarray]:
        base_actions = self._base_actions(observations)
        actor_inputs = jnp.concatenate([observations, base_actions], axis=-1)
        residual_actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, actor_inputs
        )
        full_residual = _expand_residual_actions(
            residual_actions, self.residual_action_indices, self.action_dim
        )
        actions = _compose_actions(
            base_actions, full_residual, self.residual_alpha, self.action_min, self.action_max
        )
        return self.replace(rng=new_rng), np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        base_actions = self._base_actions(observations)
        actor_inputs = jnp.concatenate([observations, base_actions], axis=-1)
        residual_actions = _eval_actions(self.actor.apply_fn, self.actor.params, actor_inputs)
        full_residual = _expand_residual_actions(
            residual_actions, self.residual_action_indices, self.action_dim
        )
        actions = _compose_actions(
            base_actions, full_residual, self.residual_alpha, self.action_min, self.action_max
        )
        return np.asarray(actions)

    def save(self, path: "str | Path") -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(serialization.to_bytes(self))

    @staticmethod
    def load(path: "str | Path", template: "ResidualSAC") -> "ResidualSAC":
        from pathlib import Path
        with Path(path).open("rb") as f:
            return serialization.from_bytes(template, f.read())
