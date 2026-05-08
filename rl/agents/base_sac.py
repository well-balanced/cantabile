from pathlib import Path
from typing import Any, Optional

import jax
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


class BaseSAC(struct.PyTreeNode):
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
    ) -> "BaseSAC":
        action_dim = spec.action.shape[-1]
        observations = zeros_like(spec.observation)
        actions = zeros_like(spec.action)
        target_entropy = config.target_entropy or -0.5 * action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = TanhNormal(
            partial(MLP, hidden_dims=config.hidden_dims, activation=getattr(nn, config.activation), activate_final=True),
            action_dim,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(config.actor_lr))

        critic_def, target_def = make_critic(config)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        no_op = optax.GradientTransformation(lambda _: None, lambda _: None)
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=optax.adam(config.critic_lr))
        target_critic = TrainState.create(apply_fn=target_def.apply, params=critic_params, tx=no_op)

        return BaseSAC(
            actor=actor, rng=rng, critic=critic, target_critic=target_critic,
            temp=make_temperature(config, temp_key),
            target_entropy=target_entropy, tau=config.tau, discount=discount,
            num_qs=config.num_qs, num_min_qs=config.num_min_qs,
            backup_entropy=config.backup_entropy,
        )

    def update_actor(self, transitions: Transition) -> tuple["BaseSAC", LogDict]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, transitions.state)
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            qs = self.critic.apply_fn({"params": self.critic.params}, transitions.state, actions, True, rngs={"dropout": key2})
            q = qs.mean(axis=0)
            actor_loss = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        return self.replace(actor=self.actor.apply_gradients(grads=grads), rng=rng), actor_info

    def update_temperature(self, entropy: float) -> tuple["BaseSAC", LogDict]:
        def temp_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            loss = temperature * (entropy - self.target_entropy).mean()
            return loss, {"temperature": temperature, "temperature_loss": loss}

        grads, temp_info = jax.grad(temp_loss_fn, has_aux=True)(self.temp.params)
        return self.replace(temp=self.temp.apply_gradients(grads=grads)), temp_info

    def update_critic(self, transitions: Transition) -> tuple["BaseSAC", LogDict]:
        dist = self.actor.apply_fn({"params": self.actor.params}, transitions.next_state)
        rng = self.rng
        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

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
    def update(self, transitions: Transition) -> tuple["BaseSAC", LogDict]:
        new_agent, critic_info = self.update_critic(transitions)
        new_agent, actor_info = new_agent.update_actor(transitions)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        return new_agent, {**actor_info, **critic_info, **temp_info}

    def sample_actions(self, observations: np.ndarray) -> tuple["BaseSAC", np.ndarray]:
        actions, new_rng = sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations)
        return self.replace(rng=new_rng), np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(eval_actions(self.actor.apply_fn, self.actor.params, observations))

    def save(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(serialization.to_bytes(self))

    @staticmethod
    def load(path: "str | Path", template: "BaseSAC") -> "BaseSAC":
        with Path(path).open("rb") as f:
            return serialization.from_bytes(template, f.read())
