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
    sample_actions, eval_actions, split_entropy_backup,
    dense_critic_obs, event_critic_obs,
    make_critic, make_temperature,
)


class DenseEventSAC(struct.PyTreeNode):
    """SAC with separate dense and event critic heads.

    dense_critic learns from per-step (dense) rewards.
    event_critic learns from event-based (sparse) rewards.
    Actor optimises the sum of both Q-values.
    """

    actor: TrainState
    rng: Any
    dense_critic: TrainState
    target_dense_critic: TrainState
    event_critic: TrainState
    target_event_critic: TrainState
    temp: TrainState
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    backup_entropy_event: bool = struct.field(pytree_node=False)

    @staticmethod
    def initialize(
        spec: EnvironmentSpec,
        config: SACConfig,
        seed: int = 0,
        discount: float = 0.99,
    ) -> "DenseEventSAC":
        action_dim = spec.action.shape[-1]
        observations = zeros_like(spec.observation)
        actions = zeros_like(spec.action)
        target_entropy = config.target_entropy or -0.5 * action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, dense_key, event_key, temp_key = jax.random.split(rng, 5)

        actor_def = TanhNormal(
            partial(MLP, hidden_dims=config.hidden_dims, activation=getattr(nn, config.activation), activate_final=True),
            action_dim,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(config.actor_lr))

        critic_def, target_def = make_critic(config)
        no_op = optax.GradientTransformation(lambda _: None, lambda _: None)

        dense_params = critic_def.init(dense_key, dense_critic_obs(observations), actions)["params"]
        dense_critic = TrainState.create(apply_fn=critic_def.apply, params=dense_params, tx=optax.adam(config.critic_lr))
        target_dense_critic = TrainState.create(apply_fn=target_def.apply, params=dense_params, tx=no_op)

        event_params = critic_def.init(event_key, event_critic_obs(observations), actions)["params"]
        event_critic = TrainState.create(apply_fn=critic_def.apply, params=event_params, tx=optax.adam(config.critic_lr))
        target_event_critic = TrainState.create(apply_fn=target_def.apply, params=event_params, tx=no_op)

        return DenseEventSAC(
            actor=actor, rng=rng,
            dense_critic=dense_critic, target_dense_critic=target_dense_critic,
            event_critic=event_critic, target_event_critic=target_event_critic,
            temp=make_temperature(config, temp_key),
            target_entropy=target_entropy, tau=config.tau, discount=discount,
            num_qs=config.num_qs, num_min_qs=config.num_min_qs,
            backup_entropy=config.backup_entropy,
            backup_entropy_event=config.backup_entropy_event,
        )

    def update_actor(self, transitions: Transition) -> tuple["DenseEventSAC", LogDict]:
        key, rng = jax.random.split(self.rng)
        dense_key, event_key, rng = jax.random.split(rng, 3)

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, transitions.state)
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            dense_qs = self.dense_critic.apply_fn(
                {"params": self.dense_critic.params}, dense_critic_obs(transitions.state), actions, True,
                rngs={"dropout": dense_key},
            )
            event_qs = self.event_critic.apply_fn(
                {"params": self.event_critic.params}, event_critic_obs(transitions.state), actions, True,
                rngs={"dropout": event_key},
            )
            q_total = dense_qs.mean(axis=0) + event_qs.mean(axis=0)
            actor_loss = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q_total).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -log_probs.mean(),
                "dense_q": dense_qs.mean(),
                "event_q": event_qs.mean(),
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        return self.replace(actor=self.actor.apply_gradients(grads=grads), rng=rng), actor_info

    def update_temperature(self, entropy: float) -> tuple["DenseEventSAC", LogDict]:
        def temp_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            loss = temperature * (entropy - self.target_entropy).mean()
            return loss, {"temperature": temperature, "temperature_loss": loss}

        grads, temp_info = jax.grad(temp_loss_fn, has_aux=True)(self.temp.params)
        return self.replace(temp=self.temp.apply_gradients(grads=grads)), temp_info

    def update_critic(self, transitions: Transition) -> tuple["DenseEventSAC", LogDict]:
        dist = self.actor.apply_fn({"params": self.actor.params}, transitions.next_state)
        rng = self.rng
        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
        temperature = self.temp.apply_fn({"params": self.temp.params})

        key, rng = jax.random.split(rng)
        dense_target_params = subsample_ensemble(key=key, params=self.target_dense_critic.params, num_sample=self.num_min_qs, num_qs=self.num_qs)
        key, rng = jax.random.split(rng)
        event_target_params = subsample_ensemble(key=key, params=self.target_event_critic.params, num_sample=self.num_min_qs, num_qs=self.num_qs)

        dense_do_key, event_do_key, rng = jax.random.split(rng, 3)
        next_dense_qs = self.target_dense_critic.apply_fn(
            {"params": dense_target_params}, dense_critic_obs(transitions.next_state), next_actions, True,
            rngs={"dropout": dense_do_key},
        )
        next_event_qs = self.target_event_critic.apply_fn(
            {"params": event_target_params}, event_critic_obs(transitions.next_state), next_actions, True,
            rngs={"dropout": event_do_key},
        )

        discount_factor = self.discount * transitions.discount
        dense_target_q = transitions.reward_dense + discount_factor * next_dense_qs.min(axis=0)
        event_target_q = transitions.reward_event + discount_factor * next_event_qs.min(axis=0)

        if self.backup_entropy:
            if self.backup_entropy_event:
                entropy_backup = split_entropy_backup(temperature, next_log_probs, discount_factor, share=0.5)
                dense_target_q -= entropy_backup
                event_target_q -= entropy_backup
            else:
                dense_target_q -= split_entropy_backup(temperature, next_log_probs, discount_factor, share=1.0)

        dense_key, event_key, rng = jax.random.split(rng, 3)

        def dense_loss_fn(critic_params):
            qs = self.dense_critic.apply_fn(
                {"params": critic_params}, dense_critic_obs(transitions.state), transitions.action, True,
                rngs={"dropout": dense_key},
            )
            loss = ((qs - dense_target_q) ** 2).mean()
            return loss, {"dense_critic_loss": loss, "dense_q": qs.mean()}

        def event_loss_fn(critic_params):
            qs = self.event_critic.apply_fn(
                {"params": critic_params}, event_critic_obs(transitions.state), transitions.action, True,
                rngs={"dropout": event_key},
            )
            loss = ((qs - event_target_q) ** 2).mean()
            return loss, {"event_critic_loss": loss, "event_q": qs.mean()}

        dense_grads, dense_info = jax.grad(dense_loss_fn, has_aux=True)(self.dense_critic.params)
        event_grads, event_info = jax.grad(event_loss_fn, has_aux=True)(self.event_critic.params)
        dense_critic = self.dense_critic.apply_gradients(grads=dense_grads)
        event_critic = self.event_critic.apply_gradients(grads=event_grads)

        target_dense_critic = self.target_dense_critic.replace(
            params=optax.incremental_update(dense_critic.params, self.target_dense_critic.params, self.tau)
        )
        target_event_critic = self.target_event_critic.replace(
            params=optax.incremental_update(event_critic.params, self.target_event_critic.params, self.tau)
        )

        info = {
            **dense_info, **event_info,
            "critic_loss": dense_info["dense_critic_loss"] + event_info["event_critic_loss"],
        }
        return self.replace(
            dense_critic=dense_critic, target_dense_critic=target_dense_critic,
            event_critic=event_critic, target_event_critic=target_event_critic,
            rng=rng,
        ), info

    @jax.jit
    def update(self, transitions: Transition) -> tuple["DenseEventSAC", LogDict]:
        new_agent, critic_info = self.update_critic(transitions)
        new_agent, actor_info = new_agent.update_actor(transitions)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        return new_agent, {**actor_info, **critic_info, **temp_info}

    def sample_actions(self, observations: np.ndarray) -> tuple["DenseEventSAC", np.ndarray]:
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
    def load(path: "str | Path", template: "DenseEventSAC") -> "DenseEventSAC":
        with Path(path).open("rb") as f:
            return serialization.from_bytes(template, f.read())
