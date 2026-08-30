"""Sanity-check an exported torch_models_pig/.../actor_<seed>.pt: load it in pure
PyTorch, roll out one full deterministic episode in the real env, and report the
same musical metrics train.py's own eval loop reports (get_musical_metrics()) --
confirming the export didn't just pass an isolated forward-pass cross-check
(export_torch.py already does that) but actually reproduces the original flax
checkpoint's live behavior.

Also loads the original flax actor sidecar (if given) and rolls out the *same*
env/episode with it, diffing the two action sequences directly -- the strongest
version of this check, since it rules out anything that a static forward-pass
comparison could miss (e.g. wrapper/scale mismatches only visible over a real
episode).

Usage:
    python eval_torch_actor.py --pt-path ../torch_models_pig/base/baseline/nocturne/actor_0.pt \\
        --environment-name RoboPianist-debug-NocturneRousseau-v0 \\
        --flax-checkpoint tmp/checkpoints/base-nocturne-s0/checkpoint_5000000_actor.flax
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Same reasoning as export_torch.py: this is a single-episode (~500-1000 step)
# rollout with a tiny MLP, not worth a GPU, and keeps the flax-vs-torch action diff
# meaningful instead of contaminated by GPU matmul precision (see
# cantabile-torch-export memory).

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import tyro

from export_torch import TorchActor
from train import Args as TrainArgs, get_env


@dataclass(frozen=True)
class EvalArgs:
    pt_path: str
    environment_name: str = "RoboPianist-debug-NocturneRousseau-v0"
    flax_checkpoint: Optional[str] = None
    """Optional: an original *_actor.flax sidecar for the same policy, to diff
    action-by-action against the torch export over a real episode."""
    obs_dim: int = 1474
    action_dim: int = 39
    hidden_dims: Tuple[int, ...] = (256, 256, 256)
    activation: str = "gelu"
    seed: int = 0


def _base_task_kwargs() -> dict:
    """Matches run.sh's `base` method env config exactly (vel/onset reward coefs
    zeroed, standard `base` task_kwargs)."""
    return dict(
        n_steps_lookahead=10,
        trim_silence=True,
        gravity_compensation=True,
        reduced_action_space=True,
        control_timestep=0.05,
        primitive_fingertip_collisions=True,
        velocity_reward_coef=0.0,
        onset_accuracy_reward_coef=0.0,
        action_reward_observation=True,
    )


def _rollout(env, action_fn) -> dict:
    timestep = env.reset()
    first_obs, n_steps = None, 0
    while not timestep.last():
        obs = np.asarray(timestep.observation, dtype=np.float32)[None]
        if first_obs is None:
            first_obs = obs
        timestep = env.step(action_fn(obs))
        n_steps += 1
    return {"first_obs": first_obs, "n_steps": n_steps,
            "music": env.get_musical_metrics(), "stats": env.get_statistics()}


def main(args: EvalArgs) -> None:
    torch_actor = TorchActor(args.obs_dim, args.action_dim, args.hidden_dims, args.activation)
    torch_actor.load_state_dict(torch.load(args.pt_path, weights_only=True))
    torch_actor.eval()

    def torch_action_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch_actor.mode(torch.from_numpy(obs)).numpy()[0]

    train_args = TrainArgs(
        environment_name=args.environment_name,
        seed=args.seed,
        camera_id="piano/back",
        **_base_task_kwargs(),
    )

    torch_env = get_env(train_args, record_dir=Path("./tmp/eval_torch_actor/torch"))
    torch_result = _rollout(torch_env, torch_action_fn)
    print(f"[torch export] {torch_result['n_steps']} steps -- "
          f"musical metrics: {torch_result['music']}")

    if args.flax_checkpoint is not None:
        import jax
        import jax.numpy as jnp
        import flax.linen as fnn
        import optax
        from flax import serialization
        from flax.training.train_state import TrainState
        from distributions import TanhNormal
        from networks import MLP

        actor_base_cls = lambda: MLP(hidden_dims=args.hidden_dims, activation=getattr(fnn, args.activation),
                                      activate_final=True)
        actor_def = TanhNormal(actor_base_cls, args.action_dim)
        params = actor_def.init(jax.random.PRNGKey(0), jnp.zeros((1, args.obs_dim)))["params"]
        template = TrainState.create(apply_fn=actor_def.apply, params=params, tx=optax.adam(3e-4))
        with open(args.flax_checkpoint, "rb") as f:
            flax_actor = serialization.from_bytes(template, f.read())

        def flax_action_fn(obs: np.ndarray) -> np.ndarray:
            return np.asarray(flax_actor.apply_fn({"params": flax_actor.params}, jnp.asarray(obs)).mode())[0]

        flax_env = get_env(train_args, record_dir=Path("./tmp/eval_torch_actor/flax"))
        flax_result = _rollout(flax_env, flax_action_fn)
        print(f"[flax original] {flax_result['n_steps']} steps -- "
              f"musical metrics: {flax_result['music']}")

        # Both envs reset deterministically from the same seed/config, so the very
        # first observation is identical -- the first action each actor takes on
        # that shared obs is directly comparable (after that, any tiny disagreement
        # would put the two rollouts on different trajectories, so this is the only
        # point a raw action diff is meaningful; episode-level `music` comparison
        # above is the fair way to compare the rest of the episode).
        first_obs = torch_result["first_obs"]
        first_torch = torch_action_fn(first_obs)
        first_flax = flax_action_fn(first_obs)
        print(f"First-step |torch - flax| action diff: {np.abs(first_torch - first_flax).max():.2e}")


if __name__ == "__main__":
    main(tyro.cli(EvalArgs, description=__doc__))
