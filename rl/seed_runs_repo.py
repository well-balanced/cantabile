"""Seed the run-queue repo: create one folder per (song, method) cell to be trained.

The queue has no plan file and no database. **The directory tree is the plan**: a
folder under `main/<song>/<method>/` means "this should be trained", and what the
folder contains says how far along it is.

    main/<song>/<method>/.gitkeep                 queued — nothing has claimed it
    main/<song>/<method>/<seed>/CLAIM-<worker>    a worker holds it; mtime is the heartbeat
    main/<song>/<method>/<seed>/*.pt              done (5M/6M/7M/8M)
    main/<song>/<method>/<seed>/FAILED            crashed; a human should look

Nothing else encodes state, so there is no status field that can disagree with the
files. "Done" is decided by the checkpoints being present, which is the one thing a
half-finished upload cannot fake — provided the worker always uploads the .pt files
before removing its CLAIM marker.

Adding a song later means creating its folders; that is the whole interface. Removing
one from the study means not creating them. Arms are staged the same way: seed only
`base` now, add `vel` once the baseline pass has been read, add `vel_res` after that.

`.gitkeep` exists for a mechanical reason, not a conceptual one: HuggingFace repos are
git, and git cannot store an empty directory. A marker file is the only way to make an
empty cell exist at all. (HuggingFace's web UI does the same thing implicitly when you
"add a file" at a new path.)

## Methods

    base      v=0,   o=0     8M     the plain RoboPianist reward
    vel       v=0.2, o=0.5   8M     dynamics-aware, end to end
    vel_res   residual       3M     branches from vel's 5M checkpoint

`vel_res` needs no dependency bookkeeping. Its prerequisite is a file: the worker
takes the cell only once `main/<song>/vel/<seed>/05000000.pt` exists, and otherwise
leaves it queued for later. Creating the folder early is therefore safe.

These three names are used identically in run.sh (`--method vel`), in wandb run names
(`vel-<song>-s43`) and as the `<method>` level of the checkpoint registry.

Usage:
    python seed_runs_repo.py --methods base --dry-run
    python seed_runs_repo.py --methods base
    python seed_runs_repo.py --methods vel            # later
"""
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tyro

# The three arms. These same names are the folder names here, the `--method` values in
# run.sh, the wandb run-name prefixes, and the `<method>` level of the checkpoint
# registry -- one vocabulary end to end. Runs launched before the rename under
# `vel_aware` / `vel_aware_residual` are folded onto it by
# `export_checkpoints_hf.canonical_method`.
METHODS = ("base", "vel", "vel_res")

README = """\
# cantabile-runs

Work queue and checkpoint store for the Cantabile dynamics study. **The directory tree
is the plan** — there is no plan file and no database.

```
main/<song>/<method>/.gitkeep                 queued, unclaimed
main/<song>/<method>/<seed>/CLAIM-<worker>    a worker holds it (mtime = heartbeat)
main/<song>/<method>/<seed>/*.pt              done: 5M / 6M / 7M / 8M checkpoints
main/<song>/<method>/<seed>/FAILED            crashed, needs a human
```

A worker lists `main/`, takes any cell with no `CLAIM` and no `.pt`, writes its claim,
trains, uploads the checkpoints, and removes the claim. A claim whose file has not been
touched for 60 minutes is considered abandoned and may be taken over.

## Methods

| folder | reward | steps | notes |
|---|---|---|---|
| `base` | v=0, o=0 | 8M | plain RoboPianist reward — the baseline arm |
| `vel` | v=0.2, o=0.5 | 8M | dynamics-aware, trained end to end |
| `vel_res` | residual on `vel` | 3M | branches from `vel`'s 5M checkpoint; 5M+3M = 8M, so all three arms cost the same |

`vel_res` is only claimable once `main/<song>/vel/<seed>/05000000.pt` exists. Its
folder can be created at any time.

## Adding a song

Create `main/<song>/<method>/.gitkeep`. The folder name is the `folder` column of
`songs_pig150.csv`, which also carries the dynamics statistics every song was selected
(or not) on.

## Checkpoints

Only steps 5M–8M are kept. Earlier ones exist during training but are not uploaded —
the training curve lives in wandb, and 1M–4M across the full study would be tens of
gigabytes that nothing reads. 5M is retained because it is the branch point `vel_res`
trains from.

Weights are PyTorch `state_dict`s for a 3x256 MLP actor with `tanh`-approximate GELU;
`obs_dim` and `action_dim` vary by arm and are recorded in `manifest.csv`.
"""


@dataclass(frozen=True)
class Args:
    repo_id: str = "well-balanced/cantabile-runs"
    songs_csv: str = "songs_pig150.csv"
    methods: Tuple[str, ...] = ("base",)
    """Which arms to queue now. Stage them: base first, vel after the baseline pass
    has been read, vel_res after that."""
    only_selected: bool = True
    """Queue only rows whose `selected` column is set (the 51-song evaluation set)."""
    private: bool = False
    upload_songs_csv: bool = True
    dry_run: bool = False


def main(args: Args) -> None:
    for m in args.methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; known: {list(METHODS)}")

    csv_path = Path(args.songs_csv)
    if not csv_path.exists():
        raise SystemExit(f"song list not found: {csv_path}")
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f)
                if not args.only_selected or r.get("selected") == "yes"]
    if not rows:
        raise SystemExit("no songs selected -- check the `selected` column")

    from huggingface_hub import HfApi
    api = HfApi()

    existing = set()
    try:
        existing = {f for f in api.list_repo_files(args.repo_id, repo_type="dataset")}
        repo_exists = True
    except Exception:
        repo_exists = False

    wanted = [f"main/{r['folder']}/{m}/.gitkeep" for r in rows for m in args.methods]
    todo = [p for p in wanted if p not in existing]

    print(f"repo      : {args.repo_id} ({'exists' if repo_exists else 'WILL BE CREATED'}, "
          f"{'private' if args.private else 'PUBLIC'})")
    print(f"songs     : {len(rows)} ({'selected only' if args.only_selected else 'all'})")
    print(f"methods   : {list(args.methods)}")
    print(f"cells     : {len(wanted)} wanted, {len(wanted) - len(todo)} already there, "
          f"{len(todo)} to create")
    for p in todo[:5]:
        print(f"    + {p}")
    if len(todo) > 5:
        print(f"    ... and {len(todo) - 5} more")

    if args.dry_run:
        print("\n(dry run -- nothing created)")
        return
    if not todo and repo_exists and not args.upload_songs_csv:
        print("\nnothing to do")
        return

    from huggingface_hub import CommitOperationAdd

    if not repo_exists:
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private,
                        exist_ok=True)
        print(f"\ncreated {args.repo_id}")

    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=io.BytesIO(b""))
           for p in todo]
    if args.upload_songs_csv:
        ops.append(CommitOperationAdd(path_in_repo="songs_pig150.csv",
                                      path_or_fileobj=str(csv_path)))
        ops.append(CommitOperationAdd(path_in_repo="README.md",
                                      path_or_fileobj=README.encode()))

    # One commit for the whole batch: 51 separate commits would bury the repo history
    # and take a hundred times as long.
    api.create_commit(
        repo_id=args.repo_id, repo_type="dataset", operations=ops,
        commit_message=f"Queue {len(todo)} cell(s): {', '.join(args.methods)} "
                       f"x {len(rows)} songs",
    )
    print(f"queued {len(todo)} cell(s) -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
