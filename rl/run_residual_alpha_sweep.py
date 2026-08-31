"""Phase A of the residual reward-design study: residual_alpha x velocity_reward_coef
x base-policy-type, screened on one song and one seed.

## Why this sweep exists

Every coefficient conclusion in the project so far -- v0.2/o0.5, margin 20, lookahead
2 -- was produced by **non-residual** runs (verified against wandb: coef_screen,
coef_stageB, margin_sweep and lookahead_sweep are all residual_alpha=0). Meanwhile
every residual run ever launched sat at a single point, (v0.5, o0.5, alpha0.2). So the
residual policy's reward has never actually been tuned; its settings are an end-to-end
result carried over by assumption. This is the first sweep that varies anything about
a residual.

## What it is meant to answer

Not "which alpha is best" on its own. `alpha` bounds how much the residual may move
the action at all (`clip(base + alpha*residual)`, residual in [-1,1] => at most
+-alpha per dim), so the value of a large velocity_reward_coef is conditional on alpha
being large enough to act on it. Tuning them separately can yield "v=0.2 is best" when
that is only true at alpha=0.2.

The deliverable is therefore the **shape** of the 4x2 grid:
  - if the best v is the same at every alpha, the two are separable and can be tuned
    independently from here on (worth stating in the paper);
  - if the best v shifts with alpha, they are coupled and every future alpha change
    invalidates the coefficient choice.

Crossing that with **both base-policy types** answers a second question: does the
residual need a dynamics-aware base (`vel_aware`, trained with v0.2/o0.5), or can it
recover the same performance correcting a vanilla `base` (v0/o0)? If `base` suffices,
the two-stage pipeline collapses to one base per song, which matters a lot for the
multi-song generalization goal.

## Grid

  alpha in {0.1, 0.2, 0.3, 0.5}   (0.2 = the value every prior residual run used)
  v     in {0.2, 0.5}             (0.2 = reward-coef sweep's pick, 0.5 = residual default)
  base  in {base, vel_aware}
  = 16 jobs, one song, one seed, onset coef fixed at 0.5.

`onset_accuracy_reward_coef` is deliberately NOT swept here -- with 16 cells already,
adding it would triple the grid to answer a question (does a residual need an onset
term at all, given the frozen base already produces onsets) that is better asked as a
2-cell o in {0, 0.5} comparison once alpha and v are pinned down.

## Song / seed choice

forelise, seed 43. Constrained, not arbitrary: a residual needs a base checkpoint for
its exact (method, song, seed), and nocturne -- where every other sweep ran -- has
`base` only for seeds 0/42 and `vel_aware` only for seed 43, so it cannot supply both
base types at any shared seed. forelise/frenchminuet/prelude/reverie each have all
three seeds for both types. forelise is picked for headroom: its `vel_aware` base
already has strong timing (onset_f1 0.806) but essentially no dynamics
(velocity_correlation -0.017), so velocity improvement from the residual is easy to
attribute.

Base checkpoints must be staged first (they live across the machine backups):

    python stage_base_checkpoints.py --songs forelise --seeds 43

Usage:
    nohup ../.venv/bin/python run_residual_alpha_sweep.py \\
        > logs/run_residual_alpha_sweep.log 2>&1 &
"""
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tyro

SONG = "forelise"
ENV_NAME = "RoboPianist-debug-ForElise-v0"
SEED = 43


@dataclass(frozen=True)
class Args:
    """Defaults reproduce the original Phase A grid. Overrides exist so an extension
    (a wider alpha range, a different onset coef) runs from this same file instead of a
    near-duplicate copy -- two sweep drivers that drift apart is how the Stage B
    duplicate-launch incident happened."""

    alphas: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
    velocity_coefs: Tuple[float, ...] = (0.2, 0.5)
    onset_coefs: Tuple[float, ...] = (0.5,)
    base_types: Tuple[str, ...] = ("base", "vel_aware")
    max_steps: int = 3_000_000
    gpus: Tuple[int, ...] = (3,)
    """Which GPUs to dispatch on. Default = GPU3, the original launch."""
    slots_per_gpu: int = 4
    """Concurrent jobs per GPU. 4 is the density Stage B proved on this hardware."""
    name_prefix: str = "res_alphaA"
    """Must be unique per launch -- two drivers emitting the same run names would
    collide on wandb and on checkpoint dirs."""
    tag: str = "residual_alpha_sweep_phaseA"
    dry_run: bool = False

# 3M (Args.max_steps), not the 5M confirmation budget: this is a screening grid, and a
# residual starts from a frozen base that already plays the piece, so it converges
# earlier than an end-to-end run. Phase B re-runs survivors at 5M across songs/seeds.
BASE_STEP = 5_000_000

RESIDUAL_ACTION_MODE = "fingers_only"

# GPU3 only. At launch GPU0/1 were running repro.train (~21GB each), GPU2 train_fmt.py
# (~12.9GB, 100% util), and GPU3 only train_bc.py (~3.4GB, 2% util) -- so GPU3 had ~29GB
# free and was effectively idle. 4 concurrent SAC jobs is the density Stage B already
# proved on this hardware ({1: 3, 2: 4, 3: 4}).
N_GPUS = 4
POLL_SECONDS = 60

def common_args(max_steps: int):
    return [
    "--root-dir", "./tmp", "--warmstart-steps", "5000", "--max-steps", str(max_steps),
    "--discount", "0.8", "--agent-config.critic-dropout-rate", "0.01",
    "--agent-config.critic-layer-norm",
    "--agent-config.hidden-dims", "256", "256", "256", "--trim-silence",
    "--gravity-compensation", "--reduced-action-space", "--control-timestep", "0.05",
    "--n-steps-lookahead", "10",
    "--action-reward-observation", "--primitive-fingertip-collisions",
    "--eval-episodes", "3", "--eval-interval", "25000", "--camera-id", "piano/back",
    "--tqdm-bar", "--mode", "online", "--checkpoint-interval", str(max_steps),
]


def base_checkpoint(base_type: str) -> Path:
    return Path("./checkpoints") / base_type / SONG / f"seed{SEED}" / f"checkpoint_{BASE_STEP}.flax"


def build_jobs(args: Args):
    jobs = []
    for base_type in args.base_types:
        for alpha in args.alphas:
            for v in args.velocity_coefs:
                for o in args.onset_coefs:
                    # Name parses as <method>-<song>-s<seed>, so it lands in the
                    # registry as <prefix>-a<alpha>-v<v>[-o<o>]-<base_type>/forelise/43
                    # with no mapping (see export_checkpoints_hf.py). The o tag is
                    # omitted when a single onset coef is used, keeping the original
                    # grid's names byte-identical to what already ran.
                    otag = "" if len(args.onset_coefs) == 1 else f"-o{o}"
                    name = (f"{args.name_prefix}-a{alpha}-v{v}{otag}-{base_type}"
                            f"-{SONG}-s{SEED}")
                    extra = [
                        "--environment-name", ENV_NAME,
                        "--velocity-reward-coef", str(v),
                        "--onset-accuracy-reward-coef", str(o),
                        "--residual-alpha", str(alpha),
                        "--residual-action-mode", RESIDUAL_ACTION_MODE,
                        "--base-checkpoint", str(base_checkpoint(base_type)),
                        "--seed", str(SEED),
                    ]
                    jobs.append((name, extra))
    return jobs


def preflight(args: Args, jobs):
    """Every base checkpoint must exist before anything launches -- a residual job that
    can't find its base fails minutes in, after the queue has already dispatched."""
    missing = [str(base_checkpoint(b)) for b in args.base_types
               if not base_checkpoint(b).exists()]
    if missing:
        raise SystemExit(
            "Missing base checkpoint(s):\n  " + "\n  ".join(missing) +
            f"\n\nStage them first:\n  python stage_base_checkpoints.py "
            f"--songs {SONG} --seeds {SEED}")
    if not args.gpus or args.slots_per_gpu < 1:
        raise SystemExit("--gpus / --slots-per-gpu allow zero concurrent jobs.")
    # A run name that already has a checkpoint dir means another driver is running it
    # (or already did). Launching it again is the Stage B duplicate-launch incident:
    # two processes writing one run, 11 of which had to be kill -9'd.
    clash = [n for n, _ in jobs if Path("./tmp/checkpoints", n).exists()]
    if clash:
        raise SystemExit(
            f"{len(clash)} run name(s) already have a checkpoint dir -- another driver "
            f"is running or ran these. Use a different --name-prefix.\n  " +
            "\n  ".join(clash[:8]))


def launch(name: str, extra: list, gpu: int, args: Args) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "WANDB_DIR": "./", "MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu),
    })
    cmd = ["../.venv/bin/python", "train.py", *common_args(args.max_steps), *extra,
           "--name", name, "--tags", args.tag]
    with open(f"logs/{name}.log", "wb") as logf:
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    print(f"[launch] {name} on GPU{gpu} (pid {proc.pid})", flush=True)
    return proc


def main(args: Args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("logs", exist_ok=True)
    queue = build_jobs(args)
    preflight(args, queue)
    print(f"{len(queue)} jobs queued ({len(args.alphas)} alphas x "
          f"{len(args.velocity_coefs)} v x {len(args.onset_coefs)} o x "
          f"{len(args.base_types)} base types, {SONG} s{SEED}, {args.max_steps} steps, "
          f"gpus={list(args.gpus)}x{args.slots_per_gpu})", flush=True)
    for name, _ in queue:
        print(f"    {name}", flush=True)
    if args.dry_run:
        print("(dry run -- nothing launched)", flush=True)
        return

    running = {gpu: [] for gpu in args.gpus}

    def gpu_with_room():
        for gpu in args.gpus:
            if len(running[gpu]) < args.slots_per_gpu:
                return gpu
        return None

    while queue or any(running.values()):
        for gpu in list(running):
            still = []
            for p in running[gpu]:
                if p.poll() is None:
                    still.append(p)
                else:
                    print(f"[done] pid {p.pid} on GPU{gpu} exited (code {p.returncode})", flush=True)
            running[gpu] = still

        launched_any = True
        while queue and launched_any:
            launched_any = False
            gpu = gpu_with_room()
            if gpu is not None:
                name, extra = queue.pop(0)
                running[gpu].append(launch(name, extra, gpu, args))
                launched_any = True

        if not queue and not any(running.values()):
            break
        time.sleep(POLL_SECONDS)

    print(f"ALL_DONE: {args.name_prefix} finished.", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
