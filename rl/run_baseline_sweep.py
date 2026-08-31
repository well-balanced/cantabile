"""Arm A of the evaluation protocol: the `base` (v=0, o=0) policy on the 51-song
evaluation set, one seed, 8M steps.

## Why one seed first

This doubles as a triage pass. The primary velocity metric is

    rwmae = onset_recall * matched_mae + (1 - onset_recall) * v_bar

so when onset_recall is low, most of the metric is the miss penalty and it reports
timing rather than dynamics. On GolliwoggsCakewalk the 8M base reached onset_recall
0.59 — already 41% miss-dominated. A song the base cannot play is a song where no
velocity method can be told apart from any other, so it is worth knowing which of the
51 those are *before* committing three arms x three seeds to all of them.

Nothing here is throwaway: this runs at the protocol's own 8M budget with the
protocol's seed, so every surviving song's run is arm A seed 43, already done.

## Song set

`songs_eval51.csv`, built from the PIG corpus (150 pieces) by two rules:

  ground-truth velocity std >= 10        — excludes constant and near-constant scores
  no note with velocity <= 4             — excludes scores whose spread comes from
                                           inaudible notes

The second rule is measured, not assumed: rendering a scale through the project's own
TimGM6mb.sf2 puts velocity 4 at -53 dBFS (amplitude 1/450 of velocity 100) and
velocity 1 at -59 dBFS, while velocity 16 is a clearly audible -32 dBFS. Four pieces
are excluded by it, three of them substantially — PianoSonataNo41StMov is 23% notes at
velocity <= 4, and PianoSonataNo211StMov's rank-1 std of 33.58 drops to 18.74 once its
51 velocity-1 notes are removed, i.e. it was never the most dynamic piece in the set.

## Reading the result

Rank the 51 by onset_recall. Songs at the bottom are candidates to drop from the
protocol — not because the method fails on them, but because the metric cannot measure
what the protocol is about. Decide the cut from the actual distribution rather than a
threshold picked in advance.

Usage:
    nohup ../.venv/bin/python run_baseline_sweep.py --gpus 0 2 \\
        > logs/run_baseline_sweep.log 2>&1 &

    # second driver on a disjoint slice once other GPUs free up
    nohup ../.venv/bin/python run_baseline_sweep.py --gpus 1 3 --song-slice 32:51 \\
        > logs/run_baseline_sweep_b.log 2>&1 &
"""
import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tyro

BASE_STEP = 8_000_000


@dataclass(frozen=True)
class Args:
    songs_csv: str = "songs_eval51.csv"
    song_slice: str = ":"
    """`start:end` over the CSV order. Two drivers must use disjoint slices --
    the started-run guard below only catches runs that already have a checkpoint
    directory, so it cannot stop two drivers from picking the same queued song."""
    seed: int = 43
    max_steps: int = BASE_STEP
    checkpoint_interval: int = 1_000_000
    """1M so the 5M checkpoint exists. Arm A doesn't branch, but arm B will, and
    keeping one interval across arms means one retention rule at export time."""
    gpus: Tuple[int, ...] = (0, 2)
    slots_per_gpu: int = 4
    name_prefix: str = "base"
    tag: str = "eval51_armA"
    poll_seconds: int = 60
    dry_run: bool = False


def common_args(max_steps: int, ckpt_interval: int):
    return [
        "--root-dir", "./tmp", "--warmstart-steps", "5000", "--max-steps", str(max_steps),
        "--discount", "0.8", "--agent-config.critic-dropout-rate", "0.01",
        "--agent-config.critic-layer-norm",
        "--agent-config.hidden-dims", "256", "256", "256", "--trim-silence",
        "--gravity-compensation", "--reduced-action-space", "--control-timestep", "0.05",
        "--n-steps-lookahead", "10",
        "--action-reward-observation", "--primitive-fingertip-collisions",
        "--eval-episodes", "3", "--eval-interval", "25000", "--camera-id", "piano/back",
        "--tqdm-bar", "--mode", "online",
        "--checkpoint-interval", str(ckpt_interval),
        # v=0, o=0: this is the plain RoboPianist reward, the protocol's baseline arm.
        "--velocity-reward-coef", "0.0", "--onset-accuracy-reward-coef", "0.0",
    ]


def load_songs(args: Args):
    with open(args.songs_csv) as f:
        rows = list(csv.DictReader(f))
    start, _, end = args.song_slice.partition(":")
    rows = rows[int(start) if start else None: int(end) if end else None]
    return [(r["song"], r["env"]) for r in rows]


def build_jobs(args: Args):
    jobs = []
    for song, env in load_songs(args):
        # Lowercased song name as the alias: unique, mechanical, and it round-trips
        # through export_checkpoints_hf.py's <method>-<song>-s<seed> parser without
        # needing 51 hand-written run.sh cases.
        alias = song.lower()
        name = f"{args.name_prefix}-{alias}-s{args.seed}"
        jobs.append((name, ["--environment-name", env, "--seed", str(args.seed)]))
    return jobs


def preflight(args: Args, jobs):
    if not Path(args.songs_csv).exists():
        raise SystemExit(f"song list not found: {args.songs_csv}")
    if not args.gpus or args.slots_per_gpu < 1:
        raise SystemExit("--gpus / --slots-per-gpu allow zero concurrent jobs.")
    clash = [n for n, _ in jobs if Path("./tmp/checkpoints", n).exists()]
    if clash:
        raise SystemExit(
            f"{len(clash)} run(s) already have a checkpoint dir — another driver is "
            f"running or ran these:\n  " + "\n  ".join(clash[:8]))


def launch(name: str, extra: list, gpu: int, args: Args) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({
        "WANDB_DIR": "./", "MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu),
    })
    cmd = ["../.venv/bin/python", "train.py",
           *common_args(args.max_steps, args.checkpoint_interval), *extra,
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
    print(f"{len(queue)} jobs queued (slice {args.song_slice}, seed {args.seed}, "
          f"{args.max_steps} steps, gpus={list(args.gpus)}x{args.slots_per_gpu})", flush=True)
    for name, _ in queue:
        print(f"    {name}", flush=True)
    if args.dry_run:
        print("(dry run -- nothing launched)", flush=True)
        return

    running = {g: [] for g in args.gpus}

    def gpu_with_room():
        for g in args.gpus:
            if len(running[g]) < args.slots_per_gpu:
                return g
        return None

    while queue or any(running.values()):
        for g in list(running):
            still = []
            for p in running[g]:
                if p.poll() is None:
                    still.append(p)
                else:
                    print(f"[done] pid {p.pid} on GPU{g} exited (code {p.returncode})", flush=True)
            running[g] = still

        launched = True
        while queue and launched:
            launched = False
            g = gpu_with_room()
            if g is not None:
                name, extra = queue.pop(0)
                running[g].append(launch(name, extra, g, args))
                launched = True

        if not queue and not any(running.values()):
            break
        time.sleep(args.poll_seconds)

    print(f"ALL_DONE: {args.tag} slice {args.song_slice} finished.", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
