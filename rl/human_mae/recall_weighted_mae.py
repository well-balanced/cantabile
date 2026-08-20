"""Recall-weighted (full-onset) velocity MAE -- comparable to the human-human anchor.

Motivation
----------
Vel MAE in the paper's tables is computed only over *matched* onsets
(env/robopianist/wrappers/evaluation.py's `velocity_mae`). If a policy misses
a systematically harder subset of onsets (dense chords, fast passages), the
reported MAE is optimistic -- it never has to answer for the notes it failed
to strike. The human-human anchor (../human_mae/compute_human_mae.py) is
computed over essentially all played notes in each MAESTRO performance, so
comparing raw Vel MAE against it directly is not apples-to-apples.

This script converts matched-only MAE into a recall-weighted MAE by charging
a fixed penalty for every *unmatched* ground-truth onset. We reuse the
penalty the paper's own Dynamics Score already treats as "as bad as playing
nothing": v_bar, the mean ground-truth velocity (main.tex, Eq.
dynamics-score). So:

    MAE_rw = R_onset * MAE_matched + (1 - R_onset) * v_bar

where R_onset = N_hit / N_gt is onset recall (`eval/onset_hit_rate`). This is
in the same 0-127 MIDI-velocity units as the human-human anchor in
results/human_mae_summary.json, so the two can be compared directly.

Data sources (priority order)
------------------------------
1. Per-run JSONs from `rl/analysis/fetch_wandb.py` (gitignored, requires
   wandb auth) -- gives exact per-song, per-seed `onset_hit_rate` and
   `velocity_mae`. Run `python rl/analysis/fetch_wandb.py` first (make sure
   its METRICS list includes 'eval/onset_hit_rate').
2. Fallback: the published, song-averaged numbers from the paper's Component
   Ablation Table (ref/writing/paper/main.tex, appendix `app:ablation-table`).
   That table reports onset_f1, not recall separately, so recall is
   *approximated* by onset_f1 (exact only when precision == recall). Always
   labeled 'approx' in the output -- rerun with real wandb data before
   putting numbers in the paper.

Usage
-----
    .venv/bin/python rl/human_mae/recall_weighted_mae.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

SONGS = ["twinkle", "clair", "nocturne"]
SEEDS = [42, 43, 44]
CONDITIONS = ["Baseline", "Vel-Aware", "Base+Res", "Vel-Aware+Res"]
VBAR = {"twinkle": 39.0, "clair": 43.8, "nocturne": 43.0}
VBAR_MEAN = float(np.mean(list(VBAR.values())))

# Published, song-averaged numbers (main.tex, appendix "Component Ablation
# Table" / app:ablation-table). Recall is APPROXIMATED by onset_f1 here --
# only exact when precision == recall. Used only when no wandb data is found.
PUBLISHED_FALLBACK = {
    "Baseline":      {"onset_f1": 0.687, "velocity_mae": 26.2},
    "Vel-Aware":     {"onset_f1": 0.781, "velocity_mae": 12.9},
    "Base+Res":      {"onset_f1": 0.711, "velocity_mae": 10.6},
    "Vel-Aware+Res": {"onset_f1": 0.770, "velocity_mae": 7.0},
}


def ci95(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    if len(v) == 1:
        return v[0], None
    mean = np.mean(v)
    se = scipy_stats.sem(v)
    t = scipy_stats.t.ppf(0.975, df=len(v) - 1)
    return mean, t * se


def rw_mae(recall, mae_matched, vbar):
    return recall * mae_matched + (1 - recall) * vbar


def load_wandb_runs(wandb_data_dir: Path):
    """Returns {condition: {song: {seed: entry}}} from fetch_wandb.py's
    per-run JSONs, or None if that directory doesn't exist / has nothing
    usable (e.g. no wandb access in this environment)."""
    if not wandb_data_dir.exists():
        return None
    data = {c: {s: {} for s in SONGS} for c in CONDITIONS}
    found_any = False
    for cond in CONDITIONS:
        tag_safe = cond.replace(" ", "_").replace("+", "p")
        for song in SONGS:
            for seed in SEEDS:
                seed_dir = wandb_data_dir / song / f"s{seed}"
                for suffix in ("", "_INCOMPLETE"):
                    fp = seed_dir / f"{tag_safe}{suffix}.json"
                    if fp.exists():
                        with open(fp) as f:
                            entry = json.load(f)
                        if entry.get("velocity_mae") is not None and entry.get("onset_hit_rate") is not None:
                            data[cond][song][seed] = entry
                            found_any = True
                        break
    return data if found_any else None


def compute_from_wandb(data):
    """Exact computation using real per-(condition, song, seed) onset_hit_rate."""
    rows = []
    for cond in CONDITIONS:
        rw_vals, matched_vals = [], []
        for song in SONGS:
            for entry in data[cond][song].values():
                rw_vals.append(rw_mae(entry["onset_hit_rate"], entry["velocity_mae"], VBAR[song]))
                matched_vals.append(entry["velocity_mae"])
        mean, hw = ci95(rw_vals)
        mmean, _ = ci95(matched_vals)
        rows.append({
            "condition": cond,
            "n": len(rw_vals),
            "matched_only_mae": round(mmean, 3) if mmean is not None else None,
            "rw_mae_mean": round(mean, 3) if mean is not None else None,
            "rw_mae_ci95": round(hw, 3) if hw is not None else None,
            "approx": False,
            "source": "wandb (exact onset_hit_rate via rl/analysis/fetch_wandb.py)",
        })
    return rows


def compute_from_fallback():
    """Approximate computation using the paper's published, song-averaged
    numbers and onset_f1 as a recall proxy."""
    rows = []
    for cond in CONDITIONS:
        d = PUBLISHED_FALLBACK[cond]
        val = rw_mae(d["onset_f1"], d["velocity_mae"], VBAR_MEAN)
        rows.append({
            "condition": cond,
            "n": None,
            "matched_only_mae": d["velocity_mae"],
            "rw_mae_mean": round(val, 3),
            "rw_mae_ci95": None,
            "approx": True,
            "source": "paper Table (approx: recall proxied by onset_f1, see docstring)",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-data-dir", default="rl/tmp/wandb_data",
                     help="Per-run JSONs written by rl/analysis/fetch_wandb.py.")
    ap.add_argument("--human-mae-summary", default="rl/human_mae/results/human_mae_summary.json")
    ap.add_argument("--out", default="rl/human_mae/results/recall_weighted_comparison.json")
    args = ap.parse_args()

    wandb_data = load_wandb_runs(Path(args.wandb_data_dir))
    if wandb_data is not None:
        print("Using exact per-run onset_hit_rate from rl/analysis/fetch_wandb.py output.")
        rows = compute_from_wandb(wandb_data)
    else:
        print(f"No wandb data at {args.wandb_data_dir} -- falling back to published, song-averaged "
              f"numbers with onset_f1 as a recall APPROXIMATION.")
        print("Run `python rl/analysis/fetch_wandb.py` (with eval/onset_hit_rate in its METRICS list, "
              "already added) first, then rerun this script, for exact numbers.\n")
        rows = compute_from_fallback()

    with open(args.human_mae_summary) as f:
        human_summary = json.load(f)
    human_anchor = human_summary["per_pair_mae_mean"]

    print(f"{'Condition':<18} {'MAE (matched-only)':<20} {'MAE (recall-weighted)':<26} approx?")
    for row in rows:
        adj = row["rw_mae_mean"]
        adj_str = f"{adj:.1f}" + (f" +/- {row['rw_mae_ci95']:.1f}" if row["rw_mae_ci95"] else "")
        matched_str = f"{row['matched_only_mae']:.1f}" if row["matched_only_mae"] is not None else "n/a"
        print(f"{row['condition']:<18} {matched_str:<20} {adj_str:<26} {row['approx']}")
    print(f"{'Human-human anchor':<18} {'--':<20} {human_anchor:<26.1f} (MAESTRO, exact)")

    out = {
        "rows": rows,
        "human_anchor_mae": human_anchor,
        "formula": "MAE_rw = R_onset * MAE_matched + (1 - R_onset) * v_bar",
        "used_wandb_data": wandb_data is not None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
