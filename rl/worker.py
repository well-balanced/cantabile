"""A training worker: claim a cell from the run-queue repo, train it, upload the
checkpoints, repeat.

Workers share nothing locally. All coordination goes through the queue repo, which is
what lets a machine join or leave without telling anyone — start one per GPU slot on
any box that can reach HuggingFace.

## The queue is a directory tree

    main/<song>/<method>/.gitkeep                 queued, unclaimed
    main/<song>/<method>/<seed>/CLAIM-<worker>    held; the file's commit time is the heartbeat
    main/<song>/<method>/<seed>/*.pt              done (5M/6M/7M/8M)
    main/<song>/<method>/<seed>/FAILED            crashed, needs a human

State is nothing but file presence, so there is no status field that can disagree with
the artifacts. Done means the checkpoints are there, which a half-finished upload
cannot fake — as long as the upload precedes the claim's removal, which it does below.

## Claiming without a lock

HuggingFace gives no compare-and-swap, so the claim is optimistic and verified by
reading back:

    write CLAIM-<me>  ->  re-read the folder  ->  is mine the only claim?

Commits to a repo are serialized, so if two workers write at once both claims end up
present and both see two; the tie is broken deterministically by sorting the claim
names, and the loser deletes its own and moves on. Either way this happens before any
GPU time is spent. The window is a couple of seconds and this holds comfortably at the
4-16 workers this study needs; it is not a design for hundreds.

A claim whose last commit is older than --stale-minutes is treated as abandoned, which
is what stops a preempted machine from parking a cell forever.

## vel_res needs no dependency bookkeeping

Its prerequisite is a file. A `vel_res` cell is claimable only once
`main/<song>/vel/<seed>/05000000.pt` exists, so the folder can be created at any time
and simply waits.

Usage:
    python worker.py --gpu 0
    python worker.py --gpu 0 --methods base --once      # single cell, then exit
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import io
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import tyro

RUNS_REPO = "well-balanced/cantabile-runs"
KEEP_STEPS = (5_000_000, 6_000_000, 7_000_000, 8_000_000)

# The three arms. Kept identical to run.sh, the wandb run-name prefixes and the
# checkpoint registry -- one vocabulary end to end.
ARMS = {
    "base":    dict(vel=0.0, onset=0.0, alpha=0.0, steps=8_000_000, base_from=None),
    "vel":     dict(vel=0.2, onset=0.5, alpha=0.0, steps=8_000_000, base_from=None),
    "vel_res": dict(vel=0.2, onset=0.5, alpha=0.2, steps=3_000_000, base_from="vel"),
}
BRANCH_STEP = 5_000_000  # where vel_res picks up vel

_CLAIM_RE = re.compile(r"CLAIM-([0-9a-f]{8})$")


@dataclass(frozen=True)
class Args:
    gpu: int = 0
    methods: Tuple[str, ...] = tuple(ARMS)
    seed: int = 43
    runs_repo: str = RUNS_REPO
    songs_csv: str = "songs_pig150.csv"
    heartbeat_minutes: int = 10
    stale_minutes: int = 60
    """A claim older than this is considered abandoned and may be taken over."""
    idle_sleep: int = 300
    """Seconds to wait before re-polling when nothing is claimable. A worker that
    finds no work sleeps rather than exiting, so a machine left running picks up
    cells the moment the queue grows."""
    once: bool = False
    dry_run: bool = False

    # Smoke-test knobs. Defaults (None/empty) mean "use the arm spec", so a normal
    # worker never touches them. They exist because the only honest way to test the
    # claim -> train -> upload -> release cycle is to run it, and a real cell is 8M
    # steps of GPU time.
    songs: Tuple[str, ...] = ()
    """Restrict to these song folders. Empty = anything in the queue."""
    max_steps_override: Optional[int] = None
    keep_steps: Tuple[int, ...] = ()
    """Which checkpoint steps to upload. Empty = KEEP_STEPS (5M-8M)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------- queue state


def read_queue(api, repo: str):
    """{(song, method): {"seeds": {seed: {"claims": [...], "pts": n, "failed": bool}}}}

    One recursive listing rather than a walk: the whole tree is a few hundred entries,
    and a per-folder walk would be hundreds of round-trips every poll.
    """
    cells, done_pts = {}, set()
    for f in api.list_repo_files(repo, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) < 4 or parts[0] != "main":
            continue
        song, method = parts[1], parts[2]
        cell = cells.setdefault((song, method), {})
        if parts[3] == ".gitkeep":
            continue
        if len(parts) < 5:
            continue
        seed, leaf = parts[3], parts[4]
        st = cell.setdefault(seed, {"claims": [], "pts": 0, "failed": False})
        if leaf.startswith("CLAIM-"):
            st["claims"].append(leaf)
        elif leaf.endswith(".pt"):
            st["pts"] += 1
            done_pts.add((song, method, seed, leaf))
        elif leaf == "FAILED":
            st["failed"] = True
    return cells, done_pts


def last_commit_age(api, repo: str, path: str) -> Optional[timedelta]:
    """How long since `path` was last written. Used as the heartbeat: touching the
    claim file makes a commit, so commit time is liveness with nothing extra to store."""
    try:
        commits = api.list_repo_commits(repo, repo_type="dataset")
    except Exception:
        return None
    for c in commits:
        if path in (c.title or "") or path in " ".join(getattr(c, "message", "") or ""):
            return _now() - c.created_at.replace(tzinfo=timezone.utc)
    return None


def claimable(api, args: Args, cells, song: str, method: str) -> bool:
    seed = str(args.seed)
    st = cells.get((song, method), {}).get(seed)
    if st:
        if st["pts"] or st["failed"]:
            return False
        for c in st["claims"]:
            age = last_commit_age(api, args.runs_repo, f"main/{song}/{method}/{seed}/{c}")
            if age is None or age < timedelta(minutes=args.stale_minutes):
                return False  # someone alive holds it
    # vel_res waits on a file, not on bookkeeping
    src = ARMS[method]["base_from"]
    if src:
        need = cells.get((song, src), {}).get(seed, {})
        if not need.get("pts"):
            return False
    return True


# ----------------------------------------------------------------- claim / release


def _put(api, args: Args, path: str, data: bytes, msg: str):
    from huggingface_hub import CommitOperationAdd
    api.create_commit(repo_id=args.runs_repo, repo_type="dataset",
                      operations=[CommitOperationAdd(path_in_repo=path,
                                                     path_or_fileobj=io.BytesIO(data))],
                      commit_message=msg)


def _rm(api, args: Args, paths):
    from huggingface_hub import CommitOperationDelete
    if not paths:
        return
    api.create_commit(repo_id=args.runs_repo, repo_type="dataset",
                      operations=[CommitOperationDelete(path_in_repo=p) for p in paths],
                      commit_message=f"release {len(paths)} marker(s)")


def try_claim(api, args: Args, song: str, method: str, me: str) -> bool:
    """Write a claim, then re-read. Returns True only if this worker owns the cell."""
    seed = str(args.seed)
    base = f"main/{song}/{method}/{seed}"
    body = f"{socket.gethostname()} gpu{args.gpu} {_now().isoformat()}\n".encode()
    _put(api, args, f"{base}/CLAIM-{me}", body, f"claim {song}/{method}/{seed}")

    files = [f.split("/")[-1] for f in api.list_repo_files(args.runs_repo, repo_type="dataset")
             if f.startswith(base + "/")]
    claims = sorted(c for c in files if _CLAIM_RE.match(c))
    if not claims:
        return False
    # Deterministic tie-break so two racers never both proceed and never both yield.
    if claims[0] != f"CLAIM-{me}":
        _rm(api, args, [f"{base}/CLAIM-{me}"])
        return False
    # Clear any losers' leftovers so the cell has exactly one claim.
    _rm(api, args, [f"{base}/{c}" for c in claims if c != f"CLAIM-{me}"])
    return True


def heartbeat_loop(api, args: Args, path: str, stop: threading.Event, me: str):
    while not stop.wait(args.heartbeat_minutes * 60):
        try:
            _put(api, args, path, f"alive {_now().isoformat()}\n".encode(),
                 f"heartbeat {path}")
        except Exception as e:  # noqa: BLE001 -- a missed beat must not kill training
            print(f"[hb] failed: {e}", flush=True)


# ----------------------------------------------------------------- run one cell


def env_for(song: str, songs_csv: str) -> str:
    import csv
    with open(songs_csv) as f:
        for r in csv.DictReader(f):
            if r["folder"] == song:
                return r["env"]
    raise SystemExit(f"song {song!r} not in {songs_csv}")


def train(args: Args, song: str, method: str) -> str:
    spec = dict(ARMS[method])
    if args.max_steps_override:
        spec["steps"] = args.max_steps_override
    run = f"{method}-{song}-s{args.seed}"
    cmd = ["../.venv/bin/python", "train.py",
           "--root-dir", "./tmp", "--warmstart-steps", "5000",
           "--max-steps", str(spec["steps"]), "--discount", "0.8",
           "--agent-config.critic-dropout-rate", "0.01",
           "--agent-config.critic-layer-norm",
           "--agent-config.hidden-dims", "256", "256", "256",
           "--trim-silence", "--gravity-compensation", "--reduced-action-space",
           "--control-timestep", "0.05", "--n-steps-lookahead", "10",
           "--action-reward-observation", "--primitive-fingertip-collisions",
           "--eval-episodes", "3", "--eval-interval", "25000",
           "--camera-id", "piano/back", "--tqdm-bar", "--mode", "online",
           "--checkpoint-interval", "1000000",
           "--environment-name", env_for(song, args.songs_csv),
           "--velocity-reward-coef", str(spec["vel"]),
           "--onset-accuracy-reward-coef", str(spec["onset"]),
           "--seed", str(args.seed), "--name", run, "--tags", "eval51"]
    if spec["base_from"]:
        src = f"./checkpoints/{spec['base_from']}/{song}/seed{args.seed}/checkpoint_{BRANCH_STEP}.flax"
        cmd += ["--residual-alpha", str(spec["alpha"]),
                "--residual-action-mode", "fingers_only", "--base-checkpoint", src]

    env = os.environ.copy()
    env.update({"WANDB_DIR": "./", "MUJOCO_GL": "egl",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "CUDA_VISIBLE_DEVICES": str(args.gpu), "MUJOCO_EGL_DEVICE_ID": "0"})
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{run}.log", "wb") as logf:
        rc = subprocess.call(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"train.py exited {rc} (see logs/{run}.log)")
    return run


def fetch_base(api, args: Args, song: str, method: str):
    """Pull the branch checkpoint a residual arm starts from into the local tree
    run.sh and train.py expect."""
    src = ARMS[method]["base_from"]
    if not src:
        return
    from huggingface_hub import hf_hub_download
    dest = Path(f"./checkpoints/{src}/{song}/seed{args.seed}")
    dest.mkdir(parents=True, exist_ok=True)
    for name in (f"{BRANCH_STEP:08d}.pt",):
        # The registry stores torch .pt, but training needs the flax checkpoint, so the
        # queue keeps the flax sidecar alongside it for exactly this handoff.
        for remote, local in ((f"main/{song}/{src}/{args.seed}/{BRANCH_STEP:08d}.flax",
                               f"checkpoint_{BRANCH_STEP}.flax"),
                              (f"main/{song}/{src}/{args.seed}/{BRANCH_STEP:08d}_actor.flax",
                               f"checkpoint_{BRANCH_STEP}_actor.flax")):
            try:
                p = hf_hub_download(args.runs_repo, remote, repo_type="dataset")
                shutil.copy2(p, dest / local)
            except Exception:
                pass
    if not (dest / f"checkpoint_{BRANCH_STEP}.flax").exists():
        raise RuntimeError(f"branch checkpoint for {song}/{src} not downloadable")


def convert(args: Args, run: str) -> Path:
    """flax -> torch, in a subprocess pinned to JAX-on-CPU.

    This MUST NOT run in this process. `preflight` initialises JAX on the GPU to check
    the device is visible, and JAX's backend cannot be re-selected afterwards -- the
    `JAX_PLATFORMS=cpu` that export_torch sets at import time then has no effect. The
    conversion's flax-vs-torch cross-check would run its forward pass on the GPU, where
    reduced-precision matmul accumulation produces a ~1e-3 disagreement with torch's
    exact-float32 CPU matmul and trips the 1e-3 tolerance. The weights are fine; the
    check is measuring the accelerator. A fresh process gets a clean backend.
    """
    out_dir = Path("tmp/worker_export") / run
    shutil.rmtree(out_dir, ignore_errors=True)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = ["../.venv/bin/python", "export_checkpoints_hf.py",
           "--out-dir", str(out_dir), "--include", run, "--all-steps",
           "--wandb-project", ""]
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"export_checkpoints_hf.py exited {rc} for {run}")
    return out_dir


def upload(api, args: Args, song: str, method: str, run: str):
    """Convert, then upload the .pt files BEFORE releasing the claim. A reader must
    never see a released cell whose artifacts are still in flight."""
    from huggingface_hub import CommitOperationAdd

    ckpt_dir = Path("tmp/checkpoints") / run
    exported = convert(args, run)
    by_step = {int(p.stem): p for p in exported.rglob("*.pt") if p.stem.isdigit()}

    ops = []
    for step in (args.keep_steps or KEEP_STEPS):
        pt = by_step.get(step)
        if pt is None:
            continue
        ops.append(CommitOperationAdd(f"main/{song}/{method}/{args.seed}/{step:08d}.pt",
                                      str(pt)))
        # The flax branch checkpoint travels with it so a residual worker on another
        # machine can start from it without needing this machine's disk.
        if step == BRANCH_STEP and ARMS[method]["base_from"] is None:
            for suffix, remote in ((".flax", f"{step:08d}.flax"),
                                   ("_actor.flax", f"{step:08d}_actor.flax")):
                f = ckpt_dir / f"checkpoint_{step}{suffix}"
                if f.exists():
                    ops.append(CommitOperationAdd(
                        f"main/{song}/{method}/{args.seed}/{remote}", str(f)))
    if not ops:
        raise RuntimeError(f"no checkpoints to upload from {ckpt_dir}")
    api.create_commit(repo_id=args.runs_repo, repo_type="dataset", operations=ops,
                      commit_message=f"done {song}/{method}/{args.seed} ({len(ops)} files)")


# ----------------------------------------------------------------- preflight


def preflight(args: Args):
    """Everything that can fail late, checked in five seconds. Each of these otherwise
    surfaces only after a full run: a bad HF token at upload time with the checkpoints
    about to be deleted, a bad wandb key as a silent fall back to offline mode whose
    metrics nobody collects, a broken EGL setup as black frames in a completed run."""
    import jax
    from huggingface_hub import HfApi

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not set — refusing to run and log offline")
    import wandb
    wandb.Api().viewer

    api = HfApi()
    api.whoami()
    api.repo_info(args.runs_repo, repo_type="dataset")

    d = jax.devices()[0]
    if d.platform != "gpu":
        raise SystemExit(f"no GPU visible to JAX (got {d.platform})")

    from robopianist import suite
    e = suite.load(environment_name="RoboPianist-debug-TwinkleTwinkleRousseau-v0", seed=0)
    e.reset()
    print(f"preflight ok — {d}, hf={api.whoami()['name']}", flush=True)
    return api


# ----------------------------------------------------------------- main loop


def main(args: Args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    me = uuid.uuid4().hex[:8]
    api = preflight(args)
    print(f"worker {me} on GPU{args.gpu}, arms={list(args.methods)}, seed={args.seed}",
          flush=True)

    while True:
        cells, _ = read_queue(api, args.runs_repo)
        todo = [(s, m) for (s, m) in cells
                if m in args.methods
                and (not args.songs or s in args.songs)
                and claimable(api, args, cells, s, m)]
        if not todo:
            print(f"nothing claimable; sleeping {args.idle_sleep}s", flush=True)
            if args.once:
                return
            time.sleep(args.idle_sleep)
            continue

        # Random rather than first: two workers starting in the same second collide on
        # the first entry almost every time and on a random one almost never.
        song, method = random.choice(todo)
        if args.dry_run:
            print(f"[dry-run] would claim {song}/{method}/{args.seed}", flush=True)
            return
        if not try_claim(api, args, song, method, me):
            print(f"lost race for {song}/{method}", flush=True)
            continue

        claim_path = f"main/{song}/{method}/{args.seed}/CLAIM-{me}"
        stop = threading.Event()
        hb = threading.Thread(target=heartbeat_loop,
                              args=(api, args, claim_path, stop, me), daemon=True)
        hb.start()
        print(f"claimed {song}/{method}/{args.seed}", flush=True)
        try:
            fetch_base(api, args, song, method)
            run = train(args, song, method)
            upload(api, args, song, method, run)
            stop.set()
            _rm(api, args, [claim_path])
            print(f"done {song}/{method}/{args.seed}", flush=True)
        except Exception as e:  # noqa: BLE001 -- one bad cell must not end the worker
            stop.set()
            print(f"FAILED {song}/{method}/{args.seed}: {e}", flush=True)
            try:
                _put(api, args, f"main/{song}/{method}/{args.seed}/FAILED",
                     f"{_now().isoformat()} {socket.gethostname()}\n{e}\n".encode(),
                     f"failed {song}/{method}/{args.seed}")
                _rm(api, args, [claim_path])
            except Exception:
                pass
            # Deliberately not retried: a run that crashed for a real reason crashes
            # identically on the next machine, and auto-retry turns one bug into a
            # fleet-wide spin. FAILED is a stop sign for a human.
        if args.once:
            return


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
