"""Stage frozen base-policy checkpoints into the `<method>/<song>/seed<N>/` tree that
run.sh's residual methods read via `--base-checkpoint`.

Residual training needs a base checkpoint per (method, song, seed), and run.sh looks
for it at exactly:

    rl/checkpoints/<method>/<song>/seed<SEED>/checkpoint_5000000.flax

Nothing writes that tree -- training writes `rl/tmp/checkpoints/<run-name>/` -- so it
has to be populated by hand, and historically was. That hand-population is where the
project's worst data bug came from: on the 3090, `vel/clair/seed{42,43,44}` (then named
`vel_aware`) were
three byte-identical copies of the s44 checkpoint, so every clair residual run trained
off the same base while appearing to be three seeds. This script exists so that never
has to be done by hand again:

- it finds each (method, song, seed) by run name across every checkpoint root,
  including the machine backups where most base checkpoints actually live;
- it copies the full agent checkpoint AND its `_actor.flax` sidecar (train.py prefers
  the sidecar and falls back to the full checkpoint -- see `_base_actor`);
- it **verifies the staged seeds are byte-distinct** and refuses to write a group where
  two seeds share a hash, which is exactly the 3090 failure re-occurring.

Usage:
    # see what would be staged
    python stage_base_checkpoints.py --songs forelise --seeds 42 43 44 --dry-run

    # actually stage, for both base types
    python stage_base_checkpoints.py --songs forelise --seeds 42 43 44
"""
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tyro

# Where base checkpoints may live, in priority order. Same roots as
# export_checkpoints_hf.py -- most pre-August base checkpoints only exist in the
# machine backups, not in this repo's rl/tmp/checkpoints.
DEFAULT_ROOTS = (
    "./tmp/checkpoints",
    "~/wynn/dgist_backup/5090/cantabile/rl/tmp/checkpoints",
    "~/wynn/dgist_backup/3090/cantabile/rl/tmp/checkpoints",
    "~/wynn/dgist_backup/2080/cantabile/rl/tmp/checkpoints",
)


@dataclass(frozen=True)
class Args:
    songs: Tuple[str, ...]
    seeds: Tuple[int, ...] = (42, 43, 44)
    methods: Tuple[str, ...] = ("base", "vel")
    """Base-policy families to stage. `base` = v0/o0, `vel` = v0.2/o0.5. These are the
    canonical arm names used in run.sh, wandb and the checkpoint registry; runs
    launched before the rename are under `vel_aware` and need that name here."""
    step: int = 5_000_000
    roots: Tuple[str, ...] = DEFAULT_ROOTS
    out_root: str = "./checkpoints"
    overwrite: bool = False
    dry_run: bool = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_run(roots, run: str, step: int) -> Optional[Tuple[Path, Optional[Path]]]:
    """(full checkpoint, actor sidecar or None) for a run name, first root wins."""
    for root in roots:
        d = root / run
        full = d / f"checkpoint_{step}.flax"
        sidecar = d / f"checkpoint_{step}_actor.flax"
        if full.exists() or sidecar.exists():
            return (full if full.exists() else None,
                    sidecar if sidecar.exists() else None)
    return None


def main(args: Args) -> None:
    roots = [Path(r).expanduser() for r in args.roots]
    missing_roots = [r for r in roots if not r.is_dir()]
    for r in missing_roots:
        print(f"  [warn] root does not exist, skipping: {r}")
    roots = [r for r in roots if r.is_dir()]

    out_root = Path(args.out_root)
    planned, problems = [], []

    for method in args.methods:
        for song in args.songs:
            group = {}
            for seed in args.seeds:
                run = f"{method}-{song}-s{seed}"
                found = find_run(roots, run, args.step)
                if found is None:
                    problems.append(f"{run}: no checkpoint_{args.step} in any root")
                    continue
                full, sidecar = found
                # Hash whichever file actually carries the weights we compare on. The
                # sidecar is the one train.py prefers, so it's the meaningful identity;
                # fall back to the full checkpoint for runs that predate sidecars.
                group[seed] = (full, sidecar, sha256(sidecar or full))

            if not group:
                continue

            # The 3090 bug guard: two seeds resolving to identical bytes means the
            # source tree was populated by copying, and staging it would silently make
            # a multi-seed residual experiment single-seed.
            by_hash = {}
            for seed, (_, _, h) in group.items():
                by_hash.setdefault(h, []).append(seed)
            dupes = {h: s for h, s in by_hash.items() if len(s) > 1}
            if dupes:
                for h, seeds in dupes.items():
                    problems.append(
                        f"{method}/{song}: seeds {seeds} are byte-identical "
                        f"({h[:12]}...) -- refusing to stage this group")
                continue

            for seed, (full, sidecar, h) in sorted(group.items()):
                dest = out_root / method / song / f"seed{seed}"
                planned.append((dest, full, sidecar, h, f"{method}-{song}-s{seed}"))

    print(f"\n{len(planned)} checkpoint(s) to stage under {out_root}/")
    for dest, full, sidecar, h, run in planned:
        have = "full+sidecar" if (full and sidecar) else ("full" if full else "sidecar")
        exists = " [EXISTS]" if (dest / f"checkpoint_{args.step}.flax").exists() else ""
        print(f"  {dest}/  <- {run} ({have}, {h[:12]}...){exists}")
        print(f"      from {(full or sidecar).parent}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  ! {p}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    written = 0
    for dest, full, sidecar, _, _ in planned:
        target = dest / f"checkpoint_{args.step}.flax"
        if target.exists() and not args.overwrite:
            print(f"  = skip (exists): {target}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if full:
            shutil.copy2(full, target)
        if sidecar:
            shutil.copy2(sidecar, dest / f"checkpoint_{args.step}_actor.flax")
        written += 1
        print(f"  + {target}")

    print(f"\nStaged {written} checkpoint(s).")
    if problems:
        raise SystemExit(f"{len(problems)} group(s)/run(s) could not be staged -- see above")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
