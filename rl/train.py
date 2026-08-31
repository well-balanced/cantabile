from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm

import sac
import specs
import replay
from flax import serialization

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers
from robopianist.suite.tasks.piano_with_shadow_hands import StyleParams
import re


@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    video_interval: int = 200_000
    """How often to attach the eval rollout video to the wandb log, in steps.

    Deliberately much coarser than `eval_interval`: `wandb.Video` copies the mp4 into
    the run directory, so uploading one per eval kept a full 800-video, 2.3 GB copy per
    8M-step run (303 runs had accumulated 90 GB of them). The local render is unlinked
    either way; this only controls how many get kept. Must be a multiple of
    `eval_interval` to ever fire -- video logging rides along with an eval."""
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "cantabile"
    entity: str = "cantabile"
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    eval_video: bool = True
    """Render an eval rollout video. Off decouples the metrics from rendering: the
    musical metrics come from MidiEvaluationWrapper, which needs no GL context at
    all, while the video wrapper needs a working EGL setup and a soundfont. On a
    host where those cannot be installed — an unprivileged Kubernetes pod, say —
    turning this off is the difference between training and not."""
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    agent_config: sac.SACConfig = sac.SACConfig()
    restore_checkpoint: Optional[Path] = None
    checkpoint_interval: int = 2_000_000
    velocity_reward_coef: float = 0.0
    onset_accuracy_reward_coef: float = 0.0
    onset_hit_bonus: float = 1.0
    velocity_reward_margin: float = 20.0
    n_steps_velocity_lookahead: int = 2
    # Residual RL options.
    base_checkpoint: Optional[Path] = None
    residual_alpha: float = 0.0
    residual_action_mode: str = "fingers_only"
    # Velocity style transform (identity by default).
    style_velocity_scale: float = 1.0
    style_velocity_bias: float = 0.0
    style_velocity_contrast: float = 1.0
    style_dynamic_trend: float = 0.0

def _action_names(action_spec) -> tuple:
    if action_spec.name:
        names = tuple(action_spec.name.split("\t"))
        if len(names) == action_spec.shape[-1]:
            return names
    return tuple(str(i) for i in range(action_spec.shape[-1]))


def _resolve_residual_action_indices(action_names: tuple, mode: str) -> tuple:
    if mode == "all":
        indices = tuple(range(len(action_names)))
    elif mode == "fingers_only":
        blocked = ("wrj", "forearm", "sustain")
        indices = tuple(
            i for i, name in enumerate(action_names)
            if not any(tok in name.lower() for tok in blocked)
        )
    elif mode == "fingers_wrist":
        blocked = ("forearm", "sustain")
        indices = tuple(
            i for i, name in enumerate(action_names)
            if not any(tok in name.lower() for tok in blocked)
        )
    else:
        raise ValueError(f"Unsupported residual_action_mode: {mode}")
    if not indices:
        raise ValueError("Residual action selection removed every action.")
    return indices


def _make_base_actor(spec: specs.EnvironmentSpec, args: Args) -> sac.TrainState:
    template = sac.SAC.initialize(spec=spec, config=args.agent_config, seed=args.seed, discount=args.discount)
    if args.base_checkpoint is None:
        return template.actor
    sidecar = Path(str(args.base_checkpoint).replace(".flax", "_actor.flax"))
    if sidecar.exists():
        with sidecar.open("rb") as f:
            actor = serialization.from_bytes(template.actor, f.read())
        print(f"Loaded base actor (sidecar) from {sidecar}")
        return actor
    base = sac.SAC.load(args.base_checkpoint, template)
    print(f"Loaded base actor from {args.base_checkpoint}")
    return base.actor


def _restore_step(path: Optional[Path]) -> int:
    if path is None:
        return 0
    match = re.search(r"checkpoint_(\d+)\.flax$", str(path))
    if match is None:
        return 0
    return int(match.group(1))


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def prefix_velocity_trajectory(row: dict) -> dict:
    result = {"eval_velocity/episode_timestep": row["episode_timestep"]}
    for key, value in row.items():
        if key != "episode_timestep":
            result[f"eval_velocity/{key}"] = value
    return result


def get_env(args: Args, record_dir: Optional[Path] = None):
    env = suite.load(
        environment_name=args.environment_name,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
            velocity_reward_coef=args.velocity_reward_coef,
            onset_accuracy_reward_coef=args.onset_accuracy_reward_coef,
            onset_hit_bonus=args.onset_hit_bonus,
            velocity_reward_margin=args.velocity_reward_margin,
            n_steps_velocity_lookahead=args.n_steps_velocity_lookahead,
            style_params=StyleParams(
                velocity_scale=args.style_velocity_scale,
                velocity_bias=args.style_velocity_bias,
                velocity_contrast=args.style_velocity_contrast,
                dynamic_trend=args.style_dynamic_trend,
            ),
        ),
    )
    if record_dir is not None:
        # The video wrapper is optional; the evaluation wrappers below are not. They
        # used to sit inside one branch, which meant a host that could not render also
        # could not measure.
        if args.eval_video:
            env = robopianist_wrappers.PianoSoundVideoWrapper(
                environment=env,
                record_dir=record_dir,
                record_every=args.record_every,
                camera_id=args.camera_id,
                height=args.record_resolution[0],
                width=args.record_resolution[1],
            )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.environment_name}-{args.seed}-{time.time()}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / "video" / run_name
    eval_dir = experiment_dir / "eval"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")
    wandb.define_metric("eval_velocity/episode_timestep")
    wandb.define_metric(
        "eval_velocity/right_hand/*",
        step_metric="eval_velocity/episode_timestep",
    )
    wandb.define_metric(
        "eval_velocity/left_hand/*",
        step_metric="eval_velocity/episode_timestep",
    )

    env = get_env(args)
    eval_env = get_env(args, record_dir=eval_dir)

    spec = specs.EnvironmentSpec.make(env)

    checkpoint_dir = Path(args.root_dir) / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.residual_alpha > 0:
        residual_action_indices = _resolve_residual_action_indices(
            _action_names(spec.action), args.residual_action_mode
        )
        print(f"Residual action mode: {args.residual_action_mode} ({len(residual_action_indices)} dims)")
        base_actor = _make_base_actor(spec, args)
        agent = sac.ResidualSAC.initialize(
            spec=spec,
            config=args.agent_config,
            base_actor=base_actor,
            seed=args.seed,
            discount=args.discount,
            residual_alpha=args.residual_alpha,
            residual_action_indices=residual_action_indices,
        )
    else:
        agent = sac.SAC.initialize(
            spec=spec,
            config=args.agent_config,
            seed=args.seed,
            discount=args.discount,
        )

    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )

    timestep = env.reset()
    replay_buffer.insert(timestep, None)
    
    start_step = _restore_step(args.restore_checkpoint)
    start_time = time.time()
    for i in tqdm(range(start_step + 1, args.max_steps + 1), disable=not args.tqdm_bar):
        # Act.
        if i < args.warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)

        # Observe.
        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        # Reset episode.
        if timestep.last():
            wandb.log({"global_step": i, **prefix_dict("train", env.get_statistics())})
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        # Train.
        if i >= args.warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % args.log_interval == 0:
                    wandb.log({"global_step": i, **prefix_dict("train", metrics)})

        # Eval.
        if i % args.eval_interval == 0:
            for _ in range(args.eval_episodes):
                timestep = eval_env.reset()
                while not timestep.last():
                    timestep = eval_env.step(agent.eval_actions(timestep.observation))
            log_dict = prefix_dict("eval", eval_env.get_statistics())
            vel_dict = prefix_dict("eval", eval_env.get_velocity_metrics())
            music_dict = prefix_dict("eval", eval_env.get_musical_metrics())
            eval_log = {"global_step": i, **log_dict, **music_dict, **vel_dict}
            if args.eval_video and args.video_interval > 0 and i % args.video_interval == 0:
                eval_log["video"] = wandb.Video(
                    str(eval_env.latest_filename), fps=4, format="mp4"
                )
            wandb.log(eval_log)
            for row in eval_env.get_velocity_trajectory_metrics():
                wandb.log({"global_step": i, **prefix_velocity_trajectory(row)})
            if args.eval_video:
                eval_env.latest_filename.unlink()

        if i % args.checkpoint_interval == 0 or i == args.max_steps:
            ckpt_path = checkpoint_dir / f"checkpoint_{i}.flax"
            agent.save(ckpt_path)
            with open(checkpoint_dir / f"checkpoint_{i}_actor.flax", "wb") as f:
                f.write(serialization.to_bytes(agent.actor))

        if i % args.log_interval == 0:
            wandb.log({"global_step": i, "train/fps": int(i / (time.time() - start_time))})


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
