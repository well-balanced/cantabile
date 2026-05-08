from pathlib import Path
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState
from functools import partial
import flax.linen as nn

from distributions import TanhNormal
from networks import MLP, subsample_ensemble
from specs import EnvironmentSpec, zeros_like
from replay import Transition
from agents._helpers import (
    LogDict, SACConfig,
    sample_actions, eval_actions,
    make_critic, make_temperature,
)


def _compose_actions(
    base: jnp.ndarray,
    residual: jnp.ndarray,
    alpha: float,
    action_min: jnp.ndarray,
    action_max: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.clip(base + alpha * residual, action_min, action_max)


def _expand_residual(
    residual_actions: jnp.ndarray,
    indices: tuple,
    full_dim: int,
) -> jnp.ndarray:
    full = jnp.zeros(residual_actions.shape[:-1] + (full_dim,), dtype=residual_actions.dtype)
    return full.at[..., jnp.asarray(indices)].set(residual_actions)


class ResidualSAC(struct.PyTreeNode):
    """SAC with a frozen base actor and a trainable residual actor.

    final_action = clip(base_action + alpha * residual_action, lo, hi)

    Residual actor input: concat(obs, base_action). Outputs actions only for
    indices in `residual_action_indices`; other dims are padded with zeros.
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

        base_actions = eval_actions(base_actor.apply_fn, base_actor.params, observations)
        actor_inputs = jnp.concatenate([observations, base_actions], axis=-1)

        actor_def = TanhNormal(
            partial(MLP, hidden_dims=config.hidden_dims, activation=getattr(nn, config.activation), activate_final=True),
            residual_action_dim,
        )
        actor_params = actor_def.init(actor_key, actor_inputs)["params"]
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(config.actor_lr))

        critic_def, target_def = make_critic(config)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        no_op = optax.GradientTransformation(lambda _: None, lambda _: None)
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(config.critic_lr))
        target_critic = TrainState.create(apply_fn=target_def.apply, params=critic_params, tx=no_op)

        return ResidualSAC(
            actor=actor, base_actor=base_actor, rng=rng,
            critic=critic, target_critic=target_critic,
            temp=make_temperature(config, temp_key),
            action_min=jnp.asarray(spec.action.minimum),
            action_max=jnp.asarray(spec.action.maximum),
            tau=config.tau, discount=discount, target_entropy=target_entropy,
            num_qs=config.num_qs, num_min_qs=config.num_min_qs,
            backup_entropy=config.backup_entropy,
            residual_alpha=residual_alpha,
            residual_action_indices=residual_action_indices,
            action_dim=full_action_dim,
        )

    def _base_actions(self, observations: jnp.ndarray) -> jnp.ndarray:
        return eval_actions(self.base_actor.apply_fn, self.base_actor.params, observations)

    def _actor_inputs(self, observations: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([observations, self._base_actions(observations)], axis=-1)

    def _composed_actions(self, observations: jnp.ndarray, residual_actions: jnp.ndarray) -> jnp.ndarray:
        base = self._base_actions(observations)
        full_residual = _expand_residual(residual_actions, self.residual_action_indices, self.action_dim)
        return _compose_actions(base, full_residual, self.residual_alpha, self.action_min, self.action_max)

    def update_actor(self, transitions: Transition) -> tuple["ResidualSAC", LogDict]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            actor_inputs = self._actor_inputs(transitions.state)
            dist = self.actor.apply_fn({"params": actor_params}, actor_inputs)
            residual_actions, log_probs = dist.sample_and_log_prob(seed=key)
            actions = self._composed_actions(transitions.state, residual_actions)
            qs = self.critic.apply_fn({"params": self.critic.params}, transitions.state, actions, True, rngs={"dropout": key2})
            q = qs.mean(axis=0)
            actor_loss = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        return self.replace(actor=self.actor.apply_gradients(grads=grads), rng=rng), actor_info

    def update_temperature(self, entropy: float) -> tuple["ResidualSAC", LogDict]:
        def temp_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            loss = temperature * (entropy - self.target_entropy).mean()
            return loss, {"temperature": temperature, "temperature_loss": loss}

        grads, temp_info = jax.grad(temp_loss_fn, has_aux=True)(self.temp.params)
        return self.replace(temp=self.temp.apply_gradients(grads=grads)), temp_info

    def update_critic(self, transitions: Transition) -> tuple["ResidualSAC", LogDict]:
        actor_inputs = self._actor_inputs(transitions.next_state)
        dist = self.actor.apply_fn({"params": self.actor.params}, actor_inputs)
        rng = self.rng
        key, rng = jax.random.split(rng)
        residual_next, next_log_probs = dist.sample_and_log_prob(seed=key)
        next_actions = self._composed_actions(transitions.next_state, residual_next)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(key=key, params=self.target_critic.params, num_sample=self.num_min_qs, num_qs=self.num_qs)
        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn({"params": target_params}, transitions.next_state, next_actions, True, rngs={"dropout": key})
        next_q = next_qs.min(axis=0)

        target_q = transitions.reward + self.discount * transitions.discount * next_q
        if self.backup_entropy:
            target_q -= self.discount * transitions.discount * self.temp.apply_fn({"params": self.temp.params}) * next_log_probs

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn({"params": critic_params}, transitions.state, transitions.action, True, rngs={"dropout": key})
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"critic_loss": loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic = self.target_critic.replace(
            params=optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        )
        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @jax.jit
    def update(self, transitions: Transition) -> tuple["ResidualSAC", LogDict]:
        new_agent, critic_info = self.update_critic(transitions)
        new_agent, actor_info = new_agent.update_actor(transitions)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        return new_agent, {**actor_info, **critic_info, **temp_info}

    def sample_actions(self, observations: np.ndarray) -> tuple["ResidualSAC", np.ndarray]:
        base_actions = self._base_actions(observations)
        actor_inputs = jnp.concatenate([observations, base_actions], axis=-1)
        residual_actions, new_rng = sample_actions(self.rng, self.actor.apply_fn, self.actor.params, actor_inputs)
        full_residual = _expand_residual(residual_actions, self.residual_action_indices, self.action_dim)
        actions = _compose_actions(base_actions, full_residual, self.residual_alpha, self.action_min, self.action_max)
        return self.replace(rng=new_rng), np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        base_actions = self._base_actions(observations)
        actor_inputs = jnp.concatenate([observations, base_actions], axis=-1)
        residual_actions = eval_actions(self.actor.apply_fn, self.actor.params, actor_inputs)
        full_residual = _expand_residual(residual_actions, self.residual_action_indices, self.action_dim)
        return np.asarray(_compose_actions(base_actions, full_residual, self.residual_alpha, self.action_min, self.action_max))

    def save(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(serialization.to_bytes(self))

    @staticmethod
    def load(path: "str | Path", template: "ResidualSAC") -> "ResidualSAC":
        with Path(path).open("rb") as f:
            return serialization.from_bytes(template, f.read())
