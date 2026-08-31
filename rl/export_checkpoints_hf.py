"""Batch-convert every flax actor checkpoint under a checkpoint root into real,
`load_state_dict()`-able PyTorch actors, lay them out in a stable content-addressed
tree, and (optionally) push that tree to a HuggingFace model repo.

This is the many-checkpoints counterpart to `export_torch.py` (which does exactly one
file, into the older `<method>/<variant>/<song>/actor_<seed>.pt` layout that mixes
"which experiment" into a hand-maintained method->bucket mapping). Here the tree is
derived mechanically from the run name, so nothing has to be mapped by hand and
nothing goes stale when a new sweep is added.

## Layout

    <collection>/<song>/<method>/<seed>/<step>.pt
        e.g. reward_coef_sweep/nocturne/coef_stageB-v0.2-o0.5/43/05000000.pt
             main/nocturne/vel_aware/43/05000000.pt
    _generalist/<method>/<seed>/<step>.pt   reserved for song-agnostic policies (BC/FMT)
    manifest.csv                            one row per .pt -- the query layer
    README.md                               layout + loading code

The design rule that makes this scale: **the directory tree is only an address, the
manifest is the index.** The path carries the five things you need to *find* a
checkpoint (collection, song, method, seed, step) and nothing else; everything you might want to
*filter or rank* by (reward coefficients, margin, lookahead, obs/action dims, wandb
id, final eval metrics, sha256) lives in `manifest.csv`. So adding a sweep axis --
say a new `residual_alpha` -- never forces a re-layout or a rename: it shows up as a
manifest column, and the new runs slot into the same four levels. Concretely:

- `<method>` is the run-name prefix **verbatim** (`coef_stageB-v0.2-o0.5`,
  `margin_sweep-m20`, `base`), not a semantic bucket. Lossless and self-describing:
  it round-trips exactly to the wandb run name, so there is no mapping to maintain
  and no way for it to drift out of sync with what was actually trained. (The
  existing `torch_models_pig/` buckets are the cautionary tale -- several of them
  have provenance nobody can now reconstruct.)
- `<step>` is zero-padded to 8 digits so lexicographic listing (the HF web file
  browser, `ls`, `sorted()`) stays in training order past 10M steps. The manifest
  keeps the unpadded integer.
- `_generalist/` is a reserved sibling namespace for policies that aren't tied to one
  song. The leading underscore keeps it from ever colliding with a real song alias.
  Nothing populates it yet -- BC/FMT generalists are 512-wide, so they're skipped by
  the architecture check below rather than silently mangled into a `TorchActor` they
  don't fit.

## Usage

    # convert everything standard-shaped into a staging dir, no upload
    python export_checkpoints_hf.py --out-dir ./tmp/hf_stage

    # same, then push to a private HF repo
    python export_checkpoints_hf.py --out-dir ./tmp/hf_stage \\
        --repo-id well-balanced/cantabile-checkpoints --private --upload

    # just one family
    python export_checkpoints_hf.py --out-dir ./tmp/hf_stage --include 'coef_stageB-v0.2-o0.5-*'
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Same reason as export_torch.py: the per-checkpoint flax-vs-torch cross-check is one
# tiny forward pass, and JAX's GPU matmuls use reduced-precision accumulation that
# produces a spurious ~1e-2 diff against torch's exact-float32 CPU matmul.

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import tyro

from export_torch import ExportArgs, TorchActor, _cross_check, _flax_to_torch_state_dict, _load_flax_actor

# Canonical song aliases. Used to disambiguate run-name parsing: a name like
# `vel_aware_residual-clair-s42-base-s44` has two `-s<digits>` groups, and only "the
# one preceded by a real song" is the training seed (the other names the base
# checkpoint's seed).
#
# Two sources, because the project has two naming eras. The short hand-written aliases
# mirror the `case $SONG in` table in run.sh; the evaluation-set songs are named by
# their lowercased score name and are read from the CSV so that adding a song to the
# study never requires editing this file. A song missing here does not fail loudly --
# its runs are silently filed as unparseable -- so the CSV is loaded eagerly and a
# missing CSV is only tolerated because older checkpoint sets predate it.
_RUN_SH_SONGS = {
    "twinkle", "clair", "nocturne", "gymnopedie", "forelise", "prelude", "waltz",
    "berceuse", "reverie", "frenchminuet", "fsuite1a", "fsuite5s", "sonatad845",
    "partita26", "etudewaltz", "bagatelle", "kreisleriana", "fsuite5g",
    "sonata23_2", "golliwogg", "sonata2_1", "sonatak279",
}

_SONG_LIST_CSV = Path(__file__).resolve().parent / "songs_eval51.csv"


def _eval_set_songs() -> set:
    if not _SONG_LIST_CSV.exists():
        return set()
    import csv as _csv
    with open(_SONG_LIST_CSV) as f:
        return {r["song"].lower() for r in _csv.DictReader(f) if r.get("song")}


_SONGS = frozenset(_RUN_SH_SONGS | _eval_set_songs())

# Checkpoint-dir tokens that don't match the canonical run.sh alias. Older runs
# predate the alias table; normalizing keeps one song from splitting into two
# top-level directories.
_SONG_ALIASES = {"clairdelune": "clair"}

# Checkpoint dirs that are not real experiments and should never be published.
_SKIP_PATTERNS = ("smoke-*", "probe*", "*-DELETEME", "* copy")

# The TorchActor architecture this exporter can faithfully represent. Anything else
# (BC/FMT generalists at 512-wide, RP1M adapters) is reported and skipped rather
# than force-fit -- see module docstring.
#
# Only the hidden widths are pinned. obs_dim and action_dim are both plain
# constructor arguments of TorchActor and both are recorded per file in the
# manifest, so a loader always knows what to build -- pinning either would drop
# whole families for a non-reason:
#   obs_dim  varies with the velocity-goal lookahead (n_steps_velocity_lookahead
#            0/1/2/10/20 -> 1298/1386/1474/2178/3058), and again for residual
#            policies, whose observation is augmented with the base action.
#   action_dim is 39 for a full-DOF policy and 30 for a fingers_only residual.
# What genuinely can't be published as a TorchActor is a different *network*:
# the 512-wide BC/FMT generalists, and the FiLM-conditioned StyleSplitFiLMMLP
# style policies (a different param tree entirely, caught by infer_shape).
_STD_HIDDEN = (256, 256, 256)
_STD_OBS_DIM = 1474  # the common case; used only for the README example
_STD_ACTION_DIM = 39  # ditto

# method -> top-level collection, first match wins. This is the ONE hand-maintained
# mapping in the layout, and it is deliberately safe to get wrong: it only groups, it
# never renames. <method> below it stays the verbatim run-name prefix, so provenance
# still round-trips even if a method lands in the "wrong" collection -- re-running with
# an edited table just moves files (the HF prune keeps the repo in sync). Without this
# level, one song's directory holds every sweep at once: nocturne alone had 82 methods.
_COLLECTIONS = (
    (r"^abl_dof", "dof_ablation"),
    (r"^margin_sweep", "margin_sweep"),
    (r"^lookahead_sweep", "lookahead_sweep"),
    (r"^coef_(screen|stageB)", "reward_coef_sweep"),
    (r"^res_alpha", "residual_alpha_sweep"),
    (r"^rp1m_pilot", "rp1m_pilot"),
    # Style-conditioning variants, tagged by the transform applied
    # (scale/contrast/trend), on either the specialist or the plain runs.
    (r"^style_specialist", "style"),
    (r"-(scale|contrast|trm|sc\d)", "style"),
    # The ad-hoc `-vXX-oYY` velocity/onset coefficient grid that predates coef_screen.
    (r"-v\d\d-o\d\d", "coef_ablation"),
)
_DEFAULT_COLLECTION = "main"

# One vocabulary for the three arms, everywhere: repo folders, run.sh, wandb, and this
# registry. The long names are what earlier runs were launched under, so they are mapped
# here rather than left to diverge -- `<method>` in a path is now always the short form.
# Applied to the leading token only, so suffixed variants come along:
# `vel_aware-v02-o05` -> `vel-v02-o05`.
#
# This does mean a path no longer reassembles into the wandb run name for pre-rename
# runs. The manifest's `run_name` column keeps the original verbatim, so provenance is
# preserved where it matters -- in the data, not in the path.
_METHOD_RENAME = {
    "vel_aware_residual": "vel_res",
    "base_residual": "base_res",
    "vel_aware": "vel",
}


def canonical_method(method: str) -> str:
    for old, new in _METHOD_RENAME.items():
        if method == old:
            return new
        if method.startswith(old + "-"):
            return new + method[len(old):]
    return method

# Collections that are classified but deliberately NOT published: abandoned lines the
# registry shouldn't carry. Dropped here rather than by deleting the mapping rules, so
# the classification stays readable and un-dropping one is a single edit. Removing a
# name from this set and re-running republishes it; the HF prune handles the reverse.
_DROP_COLLECTIONS = frozenset({"style", "rp1m_pilot"})

_SEED_RE = re.compile(r"-s(\d+)(?=-|$)")
_CKPT_RE = re.compile(r"^checkpoint_(?P<step>\d+)(?P<actor>_actor)?\.flax$")
# The `<method>/<song>/seed<N>/` tree that run.sh's --base-ckpt paths point at --
# an older, parallel layout to the flat `<run-name>/` one. Both appear in the
# machine backups, so discovery handles both.
_NESTED_SEED_RE = re.compile(r"^seed(\d+)$")


@dataclass(frozen=True)
class Args:
    out_dir: str
    """Local staging directory the tree is built in (also what gets uploaded)."""
    checkpoint_roots: Tuple[str, ...] = ("local=./tmp/checkpoints",)
    """`label=path` entries, in priority order. Checkpoints are keyed by
    (song, method, seed, step) and the FIRST root to supply a key wins, so put the
    authoritative machine first. A later root supplying the same key with different
    bytes is reported as a collision rather than silently dropped -- that would mean
    two machines trained the same config to different weights."""
    include: str = "*"
    """Glob over checkpoint *directory* names (i.e. run names)."""
    exclude: str = ""
    """Glob over run names, applied after --include. Empty = exclude nothing."""
    overwrite: bool = False
    """Re-convert checkpoints whose .pt already exists in out_dir."""
    strict_dims: bool = True
    """Skip checkpoints whose hidden widths aren't the standard TorchActor MLP.
    obs_dim and action_dim are never checked -- both vary legitimately (lookahead
    width, fingers_only residuals) and both are constructor arguments. Turning
    this off attempts every actor-shaped checkpoint; genuinely different network
    shapes then fail the strict state_dict load."""
    wandb_project: str = "cantabile/cantabile"
    """Pulled to enrich manifest.csv with config + final metrics. Empty = skip."""
    repo_id: str = ""
    private: bool = True
    upload: bool = False
    """Actually push to HF. Without this the tree is only built locally."""
    all_steps: bool = False
    """Publish every saved checkpoint of a run, not just its final step."""


def collection_of(method: str) -> str:
    """Which top-level campaign directory a method belongs to."""
    for pattern, name in _COLLECTIONS:
        if re.search(pattern, method):
            return name
    return _DEFAULT_COLLECTION


def parse_run_name(run: str) -> Optional[Tuple[str, str, int]]:
    """`coef_stageB-v0.2-o0.5-nocturne-s43` -> (method, song, seed).

    Run names are `<method>-<song>-s<seed>` with two complications the machine
    backups are full of:

    1. The method itself contains dashes (`coef_stageB-v0.2-o0.5`,
       `abl_dof-fingers`), so this can't just split on "-".
    2. `-s<digits>` can appear more than once, and a trailing tag can follow the
       seed -- `vel_aware_residual-clair-s42-base-s44` (trained on clair seed 42,
       off a base checkpoint from seed 44) and `vel_aware-nocturne-s42-scale12`.

    So: find every `-s<digits>` group and pick the one whose preceding token is a
    known song alias, rather than the first or last blindly. Anything after the
    seed becomes a variant tag folded into the method, which keeps variants
    grouped under one method level. The verbatim `run_name` is kept in the
    manifest regardless, so provenance never depends on this parse being pretty.
    """
    for m in _SEED_RE.finditer(run):
        head = run[:m.start()]
        song_raw = head.rsplit("-", 1)[-1]
        song = _SONG_ALIASES.get(song_raw, song_raw)
        if song not in _SONGS:
            continue
        method = head[: -(len(song_raw) + 1)]
        if not method:
            continue
        variant = run[m.end():].lstrip("-")
        method = f"{method}-{variant}" if variant else method
        return canonical_method(method), song, int(m.group(1))
    return None


def infer_shape(ckpt: Path) -> Optional[Tuple[int, int, Tuple[int, ...]]]:
    """Read obs_dim / action_dim / hidden_dims straight out of the checkpoint.

    Avoids a per-family hardcoded dims table -- the file already knows its own
    shape, and guessing wrong here would silently produce a garbage .pt (the
    template would restore into mismatched leaves).
    """
    from flax import serialization

    tree = serialization.msgpack_restore(ckpt.read_bytes())
    # Actor sidecars are `{"params": ...}`; full agent checkpoints nest the actor
    # alongside critic/temperature as `{"actor": {"params": ...}, ...}`.
    params = tree.get("params") or (tree.get("actor") or {}).get("params")
    if not params or "MLP_0" not in params:
        return None
    mlp = params["MLP_0"]
    dense_keys = sorted((k for k in mlp if k.startswith("Dense_")), key=lambda k: int(k.split("_")[1]))
    if not dense_keys:
        return None
    obs_dim = int(np.asarray(mlp[dense_keys[0]]["kernel"]).shape[0])
    hidden = tuple(int(np.asarray(mlp[k]["kernel"]).shape[1]) for k in dense_keys)
    try:
        action_dim = int(np.asarray(params["OutputDenseMean"]["kernel"]).shape[1])
    except KeyError:
        return None
    return obs_dim, action_dim, hidden


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_wandb_meta(project: str) -> dict:
    """{run_name: {config + final metrics}} for manifest enrichment. Best-effort:
    a missing/unauthenticated wandb must not block a local export."""
    try:
        import wandb

        runs = wandb.Api().runs(project)
    except Exception as e:  # noqa: BLE001 -- any wandb failure degrades to "no metadata"
        print(f"  [warn] wandb metadata unavailable ({type(e).__name__}: {e}); manifest will omit it")
        return {}
    meta = {}
    for r in runs:
        if r.state != "finished":
            continue
        c, s = r.config, r.summary
        # Last writer wins on duplicate names; runs are returned newest-first, so
        # take the first (newest) and don't let an older rerun clobber it.
        meta.setdefault(
            r.name,
            {
                "wandb_id": r.id,
                "env_id": c.get("environment_name"),
                "velocity_reward_coef": c.get("velocity_reward_coef"),
                "onset_accuracy_reward_coef": c.get("onset_accuracy_reward_coef"),
                "velocity_reward_margin": c.get("velocity_reward_margin"),
                "n_steps_velocity_lookahead": c.get("n_steps_velocity_lookahead"),
                "n_steps_lookahead": c.get("n_steps_lookahead"),
                "max_steps": c.get("max_steps"),
                # Prefixed `run_` and not `final_`: these are the *run's* endpoint
                # metrics, repeated on every step row of that run. They describe the
                # training run, NOT the individual checkpoint -- a 1M-step row
                # carries the run's 5M numbers. Per-checkpoint eval would need a
                # wandb history pull (one call per run) matched to the nearest
                # logged step; not done here to keep this a single cheap API sweep.
                "run_final_f1": s.get("eval/f1"),
                "run_final_onset_f1": s.get("eval/onset_f1"),
                "run_final_velocity_mae_recall_weighted": s.get("eval/velocity_mae_recall_weighted"),
                "run_final_velocity_correlation": s.get("eval/velocity_correlation"),
            },
        )
    return meta


def discover(label: str, root: Path, all_steps: bool = False):
    """Yield (run_name, method, song, seed, step, ckpt_path, full_agent, label).

    Handles both on-disk layouts that exist across the machine backups:
      flat:   <root>/<run-name>/checkpoint_<step>[_actor].flax
      nested: <root>/<method>/<song>/seed<N>/checkpoint_<step>.flax   (run.sh --base-ckpt)

    Within one run, an `_actor.flax` sidecar is preferred over the full agent
    checkpoint for the same step -- same weights, ~4x smaller to read, and no
    critic/optimizer tree to restore around them. Older runs only ever wrote the
    full checkpoint, so those fall back to it.
    """
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        flat = sorted(run_dir.glob("checkpoint_*.flax"))
        if flat:
            yield from _emit(label, run_dir.name, flat, all_steps)
            continue
        # Nested: <method>/<song>/seed<N>/
        for song_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            group = []
            for seed_dir in sorted(p for p in song_dir.iterdir() if p.is_dir()):
                sm = _NESTED_SEED_RE.match(seed_dir.name)
                if not sm:
                    continue
                ckpts = sorted(seed_dir.glob("checkpoint_*.flax"))
                if ckpts:
                    group.append((sm.group(1), ckpts))

            # These trees were in places populated by copying ONE run's checkpoint
            # into every seed folder (confirmed: 3090's vel_aware/clair/seed42,43,44
            # are byte-identical). The seed label is then a lie, and publishing it
            # would let a seed-averaged result silently count one run three times.
            # There's no way to tell which seed the shared file really came from, so
            # refuse the whole group rather than attribute it wrongly -- the genuine
            # per-seed weights, where they survive, come from a properly named
            # `<method>-<song>-s<seed>` run dir on another machine.
            digests = {}
            for seed, ckpts in group:
                digests.setdefault(sha256(ckpts[0]), []).append(seed)
            copied = {s for seeds in digests.values() if len(seeds) > 1 for s in seeds}
            for seed, ckpts in group:
                if seed in copied:
                    _UNEMITTED.append((
                        f"[{label}] {run_dir.name}/{song_dir.name}/seed{seed}",
                        "byte-identical to another seed dir in the same tree -- "
                        "seed label unreliable, refusing to attribute"))
                    continue
                yield from _emit(label, f"{run_dir.name}-{song_dir.name}-s{seed}", ckpts, all_steps)


# Runs discovered but never emitted, with the reason -- reported at the end so a
# silently-dropped directory can't masquerade as "there was nothing there".
_UNEMITTED: list = []


def _emit(label: str, run: str, ckpts, all_steps: bool = False):
    if any(fnmatch.fnmatch(run, p) for p in _SKIP_PATTERNS):
        _UNEMITTED.append((f"[{label}] {run}", "not a real experiment (skip pattern)"))
        return
    parsed = parse_run_name(run)
    if parsed is None:
        # Not exported. A run whose name carries no <song>-s<seed> is either
        # song-agnostic (the BC/FMT generalists) or from the abandoned
        # event-critic / e2e / adaptive-alpha line (`nt-*`/`tw-*`/`wz-*`/`fs<N>-*`),
        # which was deliberately deleted from the working tree and only survives in
        # the machine backups -- it is not wanted in the registry. Reported rather
        # than silently dropped so a genuinely new naming scheme is still visible.
        _UNEMITTED.append((f"[{label}] {run}",
                           "no <song>-s<seed> in name (song-agnostic or discarded line)"))
        return
    method, song, seed = parsed
    by_step = {}
    for c in ckpts:
        cm = _CKPT_RE.match(c.name)
        if not cm:
            continue
        step = int(cm["step"])
        is_actor = bool(cm["actor"])
        # Prefer the actor sidecar; only take the full agent if no sidecar exists.
        if step not in by_step or (is_actor and by_step[step][1]):
            by_step[step] = (c, not is_actor)
    # Final step only by default. The reward-coef sweep drivers pass
    # `--checkpoint-interval 1000000` (run.sh's default is 2M), so those runs wrote a
    # checkpoint every 1M steps -- 390 of the registry's files came from 98 runs that
    # way, for training-curve inspection that wandb history already covers. Publishing
    # only the endpoint keeps the registry about *trained policies*, one per run.
    steps = sorted(by_step) if all_steps else [max(by_step)] if by_step else []
    for step in steps:
        path, full_agent = by_step[step]
        yield run, method, song, seed, step, path, full_agent, label


def convert_one(ckpt: Path, out_path: Path, obs_dim: int, action_dim: int,
                hidden: Tuple[int, ...], full_agent: bool = False) -> float:
    """flax checkpoint -> TorchActor state_dict on disk. Returns cross-check max diff."""
    export_args = ExportArgs(
        checkpoint=str(ckpt), obs_dim=obs_dim, action_dim=action_dim,
        hidden_dims=hidden, activation="gelu", song="_", seed=0,
        full_agent_checkpoint=full_agent,
    )
    flax_actor = _load_flax_actor(export_args)
    state_dict = _flax_to_torch_state_dict(flax_actor.params, len(hidden))

    torch_actor = TorchActor(obs_dim, action_dim, hidden, "gelu")
    torch_actor.load_state_dict(state_dict, strict=True)

    max_diff = _cross_check(flax_actor, torch_actor, obs_dim)
    if max_diff > 1e-3:
        raise RuntimeError(f"flax/torch mismatch {max_diff:.2e} -- refusing to save {ckpt}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_actor.state_dict(), out_path)
    return max_diff


README_TEMPLATE = """---
tags:
- reinforcement-learning
- robopianist
- robotics
- piano
library_name: pytorch
---

# cantabile checkpoints

PyTorch actor weights for every SAC policy trained in the
[cantabile](https://github.com/well-balanced/cantabile) project (RoboPianist,
dynamics-aware piano performance). Converted from the JAX/Flax training checkpoints
by `rl/export_checkpoints_hf.py`; every file passed a flax-vs-torch forward-pass
cross-check (max abs diff on the pre-tanh mean, recorded per file in `manifest.csv`).

Actors only -- critic and temperature are training-only state and are not published.

## Layout

```
<collection>/<song>/<method>/<seed>/<step>.pt
_generalist/<method>/<seed>/<step>.pt   # reserved; song-agnostic policies
manifest.csv
```

- **collection** -- which campaign the run belongs to: `main` (the four `run.sh`
  methods: `base`, `vel_aware`, `base_residual`, `vel_aware_residual`),
  `reward_coef_sweep`, `margin_sweep`, `lookahead_sweep`, `dof_ablation`,
  `coef_ablation`, `residual_alpha_sweep`. Without this level a single song directory
  holds every sweep at once -- nocturne alone spans 82 methods.
- **song** -- `run.sh` song alias (`nocturne`, `clair`, `fsuite1a`, ...).
- **method** -- the wandb run-name prefix, verbatim (`base`,
  `coef_stageB-v0.2-o0.5`, `margin_sweep-m20`, `lookahead_sweep-l2`, ...). It is
  deliberately *not* a semantic bucket. Three arms carry a short canonical name --
  `base`, `vel`, `vel_res` -- used identically in the run-queue repo, `run.sh`, wandb and
  here; runs launched under the older long names (`vel_aware`, `vel_aware_residual`) are
  mapped onto it. The `run_name` column of `manifest.csv` always keeps the launched name
  verbatim, so provenance lives in the data rather than in the path.
  `collection` is the only hand-maintained grouping, and it is safe to get wrong --
  it groups, it never renames.
- **seed** -- training seed.
- **step** -- env step, zero-padded to 8 digits so listings stay in training order.

**`obs_dim` and `action_dim` are not constant across this repo.** The velocity-goal
lookahead widens the observation (`n_steps_velocity_lookahead` 0/1/2/10/20 ->
`obs_dim` 1298/1386/1474/2178/3058), and residual policies are wider still
(base-action-augmented) with `action_dim=30` (fingers_only) rather than 39. Read both
from `manifest.csv` and pass them to the constructor -- don't assume {obs_dim}/{action_dim}.

**Residual checkpoints (`is_residual == True`, methods containing `residual`) are not
runnable on their own.** They output a correction that is scaled by `residual_alpha`
and added to a *frozen base policy's* action; you need the matching base checkpoint
(same song/seed, `base` or `vel_aware` method) to roll one out.

`obs_dim` is **not** constant across this repo: the velocity-goal lookahead widens the
observation (`n_steps_velocity_lookahead` 0/1/2/10/20 -> `obs_dim`
1298/1386/1474/2178/3058). Read `obs_dim` from `manifest.csv` and pass it to the
constructor rather than assuming {obs_dim}.

The path is an **address**; `manifest.csv` is the **index**. Anything you would filter
or rank by -- reward coefficients, velocity margin, lookahead, obs/action dims, wandb
id, final eval metrics, sha256 -- is a manifest column, not a directory level. New
sweep axes become new columns and slot into the same four levels without a re-layout.

## Loading

```python
import torch, torch.nn as nn
from huggingface_hub import hf_hub_download

class TorchActor(nn.Module):
    def __init__(self, obs_dim={obs_dim}, action_dim={action_dim}, hidden_dims={hidden}):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU(approximate="tanh")]  # NOT default GELU
            d = h
        self.mlp = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(d, action_dim)
        self.log_std_layer = nn.Linear(d, action_dim)

    def forward(self, obs):
        x = self.mlp(obs)
        return self.mean_layer(x), torch.clamp(self.log_std_layer(x), -20, 2)

    def mode(self, obs):  # deterministic action
        return torch.tanh(self.forward(obs)[0])

path = hf_hub_download("{repo_id}", "reward_coef_sweep/nocturne/coef_stageB-v0.2-o0.5/43/05000000.pt")
actor = TorchActor()
actor.load_state_dict(torch.load(path, map_location="cpu"))
actor.eval()
```

`nn.GELU(approximate="tanh")` is load-bearing: flax's `nn.gelu` defaults to the tanh
approximation while torch's `nn.GELU()` defaults to the exact erf form. Using torch's
default silently changes the policy's outputs.

## Selecting a checkpoint

```python
import pandas as pd
from huggingface_hub import hf_hub_download

m = pd.read_csv(hf_hub_download("{repo_id}", "manifest.csv"))
best = (m[(m.velocity_reward_coef == 0.2) & (m.onset_accuracy_reward_coef == 0.5)]
          .sort_values("step").groupby(["song", "seed"]).tail(1))
```

## Caveats

- **`duplicate_of` -- check it before averaging over seeds.** Some
  `<method>/<song>/seed<N>/` directories on the source machines were populated by
  copying one run's checkpoint into several seed folders, so a set of paths that look
  like independent seeds can be the same weights repeated. Rows where `duplicate_of`
  is non-empty are byte-identical to the path it names. Known case: `clair/vel_aware`
  seeds 42/43/44 are one checkpoint, which also means residual runs built on that base
  share it.

- `run_final_*` columns are the **run's** endpoint metrics, repeated on every step
  row of that run -- they describe the training run, not the individual checkpoint.
  A 1M-step row carries its run's 5M numbers. Don't rank checkpoints *within* a run
  by them.
- `eval/velocity_mae_recall_weighted` and the onset precision/recall metrics were
  added part-way through the project, so `run_final_*` is genuinely empty for runs
  that predate them -- not a conversion failure.
- Seed sets are not uniform across songs (some families use 42/43/44, others
  43/44/45). Check `manifest.csv` before assuming a paired comparison.
"""


def main(args: Args) -> None:
    out_dir = Path(args.out_dir)

    roots = []
    for entry in args.checkpoint_roots:
        label, _, path = entry.partition("=")
        if not path:
            label, path = "local", entry
        p = Path(path).expanduser()
        if not p.is_dir():
            raise SystemExit(f"checkpoint root not found: {p}")
        roots.append((label, p))

    meta = fetch_wandb_meta(args.wandb_project) if args.wandb_project else {}
    if meta:
        print(f"wandb metadata for {len(meta)} finished runs")

    rows, skipped, collisions = [], [], []
    seen = {}  # (song, method, seed, step) -> source label of the winning root
    for label, root in roots:
        print(f"\nScanning [{label}] {root} ...")
        for run, method, song, seed, step, ckpt, full_agent, _ in discover(label, root, args.all_steps):
            if not fnmatch.fnmatch(run, args.include):
                continue
            if args.exclude and fnmatch.fnmatch(run, args.exclude):
                skipped.append((f"[{label}] {run}", "--exclude"))
                continue

            key = (song, method, seed, step)
            if key in seen:
                collisions.append((f"[{label}] {run}/{ckpt.name}", seen[key]))
                continue

            shape = infer_shape(ckpt)
            if shape is None:
                skipped.append((f"[{label}] {run}/{ckpt.name}", "no MLP_0/OutputDenseMean params"))
                continue
            obs_dim, action_dim, hidden = shape
            if args.strict_dims and hidden != _STD_HIDDEN:
                skipped.append((f"[{label}] {run}/{ckpt.name}",
                                f"nonstandard architecture {obs_dim}/{action_dim}/{hidden}"))
                continue

            collection = collection_of(method)
            if collection in _DROP_COLLECTIONS:
                skipped.append((f"[{label}] {run}/{ckpt.name}",
                                f"collection '{collection}' is not published"))
                continue
            out_path = out_dir / collection / song / method / str(seed) / f"{step:08d}.pt"
            if out_path.exists() and not args.overwrite:
                max_diff = float("nan")
            else:
                try:
                    max_diff = convert_one(ckpt, out_path, obs_dim, action_dim, hidden, full_agent)
                except Exception as e:  # noqa: BLE001 -- one bad checkpoint must not abort the sweep
                    skipped.append((f"[{label}] {run}/{ckpt.name}", f"{type(e).__name__}: {e}"))
                    continue
                print(f"  + {out_path.relative_to(out_dir)}  (diff {max_diff:.1e})")

            seen[key] = label
            rows.append({
                "path": str(out_path.relative_to(out_dir)),
                "collection": collection,
                "song": song,
                "method": method,
                "seed": int(seed),
                "step": step,
                "run_name": run,
                "source_machine": label,
                "source_checkpoint": str(ckpt),
                "from_full_agent_checkpoint": full_agent,
                # Residual policies output a correction added to a frozen base
                # policy's action -- the weights alone are not a runnable policy.
                "is_residual": "residual" in method,
                "obs_dim": obs_dim, "action_dim": action_dim,
                "hidden_dims": "-".join(str(h) for h in hidden),
                "activation": "gelu",
                "crosscheck_max_diff": max_diff,
                "sha256": sha256(out_path),
                **meta.get(run, {}),
            })

    if not rows:
        raise SystemExit("nothing to export -- check --include / --checkpoint-root")

    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["collection", "song", "method", "seed", "step"])

    # Flag byte-identical weights. Not merely cosmetic: the `<method>/<song>/seed<N>/`
    # trees on the machine backups were in places populated by *copying* one run's
    # checkpoint into several seed directories, so a set of paths that look like N
    # independent seeds can be one seed N times. Anything downstream that averages
    # over seeds has to know. `duplicate_of` points at the first path (in sort order)
    # carrying those exact weights; empty means this row is that canonical copy.
    canonical = df.groupby("sha256")["path"].transform("first")
    df["duplicate_of"] = canonical.where(canonical != df["path"], "")
    n_dupe = int((df.duplicate_of != "").sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "README.md").write_text(README_TEMPLATE.format(
        repo_id=args.repo_id or "well-balanced/cantabile-checkpoints",
        obs_dim=_STD_OBS_DIM, action_dim=_STD_ACTION_DIM, hidden=list(_STD_HIDDEN),
    ))

    total_mb = sum(f.stat().st_size for f in out_dir.rglob("*.pt")) / 1e6
    print(f"\n{len(df)} checkpoints -> {out_dir}  ({total_mb:.0f} MB)")
    print(f"  {df.song.nunique()} songs, {df.method.nunique()} methods, "
          f"{len(df.groupby(['song', 'method', 'seed']))} (song, method, seed) runs")
    if n_dupe:
        print(f"  !! {n_dupe} files are byte-identical to another file "
              f"(see manifest `duplicate_of`) -- some 'seeds' are the same weights")
    print("  per machine: " + ", ".join(
        f"{k}={v}" for k, v in df.source_machine.value_counts().items()))
    if collisions:
        print(f"  {len(collisions)} duplicate (song, method, seed, step) skipped "
              f"(already supplied by an earlier root):")
        for name, winner in collisions[:10]:
            print(f"    - {name} (kept [{winner}]'s copy)")
        if len(collisions) > 10:
            print(f"    ... and {len(collisions) - 10} more")
    if _UNEMITTED:
        print(f"  {len(_UNEMITTED)} run dirs not exported:")
        for name, why in _UNEMITTED:
            print(f"    - {name}: {why}")
    if skipped:
        print(f"  {len(skipped)} checkpoints skipped:")
        for name, why in skipped[:20]:
            print(f"    - {name}: {why}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    if not args.upload:
        print("\n(dry run -- pass --upload with --repo-id to push)")
        return
    if not args.repo_id:
        raise SystemExit("--upload requires --repo-id")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"\nUploading to https://huggingface.co/{args.repo_id} "
          f"({'private' if args.private else 'public'}) ...")
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Add {len(df)} actor checkpoints "
                       f"({df.song.nunique()} songs, {df.method.nunique()} methods)",
    )

    # upload_folder only adds and updates -- it never deletes. Without this, a file
    # withdrawn from the export (a checkpoint found to be mislabeled, a run renamed)
    # would linger in the repo forever, still downloadable and no longer described by
    # manifest.csv. Anything .pt in the repo that this export didn't produce is such
    # an orphan, so delete it and keep the repo exactly equal to the manifest.
    local = {str(f.relative_to(out_dir)) for f in out_dir.rglob("*.pt")}
    orphans = [f for f in api.list_repo_files(args.repo_id, repo_type="model")
               if f.endswith(".pt") and f not in local]
    if orphans:
        print(f"Pruning {len(orphans)} file(s) no longer in the export:")
        for f in orphans:
            print(f"    - {f}")
        api.delete_files(repo_id=args.repo_id, repo_type="model", delete_patterns=orphans,
                         commit_message=f"Prune {len(orphans)} withdrawn checkpoint(s)")
    print(f"Done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
