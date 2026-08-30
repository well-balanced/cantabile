"""Pulls the full wandb history (not just final summary) for the Phase 2 reward-coef
sweep -- Stage A (screening, 3M steps) + Stage B (confirmation, 5M steps) + the
v=0/o=0 baseline runs (5M steps) used as the "does f1 drop vs no reward at all"
guardrail check -- into flat CSVs for later paper figures (time-series over
global_step, not just endpoint values).

Outputs (this directory):
  stageA_history.csv   -- one row per (run, eval logging step), Stage A (screening)
  stageB_history.csv   -- one row per (run, eval logging step), Stage B (confirmation)
  baseline_history.csv -- same, for the v=0/o=0 nocturne/clair baselines
  *_summary.csv         -- one row per run (final/best values + config), for quick
                            lookups without loading the full history

Re-run any time to refresh (e.g. once Stage A's crashed duplicates or a rerun
changes wandb state) -- it always re-pulls from the API, no local caching.

Usage: ../../.venv/bin/python build_dataset.py
"""
import wandb
import pandas as pd

PROJECT = "cantabile/cantabile"

# Every eval/* metric that exists on these runs (see evaluation.py's
# get_musical_metrics / get_velocity_metrics) plus the step/return/length context.
# eval/f1 vs eval/onset_f1 are DIFFERENT metrics (frame-level key-press F1 vs
# event-level onset-detection F1) -- see README.md in this directory before using.
METRICS = [
    "global_step",
    "eval/f1", "eval/precision", "eval/recall",
    "eval/onset_f1", "eval/onset_precision", "eval/onset_recall",
    "eval/onset_hit_rate", "eval/onset_fp_rate", "eval/onset_miss_rate",
    "eval/match_rate",
    "eval/expressive_f1",
    "eval/velocity_mae", "eval/velocity_mae_recall_weighted",
    "eval/velocity_rmse", "eval/velocity_std", "eval/velocity_bias",
    "eval/velocity_correlation",
    "eval/sustain_f1", "eval/sustain_precision", "eval/sustain_recall",
    "eval/return", "eval/length",
]

SONG_BY_ENV = {
    "RoboPianist-debug-NocturneRousseau-v0": "nocturne",
    "RoboPianist-debug-ClairDeLune-v0": "clair",
}

# Per-song mean GT MIDI onset velocity (v_bar). Reconstructed from this sweep's own
# eval/velocity_mae_recall_weighted = recall*matched_mae + (1-recall)*v_bar (solved
# for v_bar) -- constant to ~1e-14 across 24 runs/song, confirming it really is a
# fixed per-song number, and it matches rl/human_mae/results/song_velocity_stats.json
# to 4 decimals. Also just fixed in eval_bc_generalist.VBAR, which had a stale
# nocturne value and no clair entry at all -- see that file's comment.
VBAR = {"nocturne": 39.6068, "clair": 44.1646}


def song_of(env_name: str) -> str:
    return SONG_BY_ENV.get(env_name, env_name)


def add_dynamics_score(df: pd.DataFrame) -> pd.DataFrame:
    """dynamics_score = onset_f1 * max(0, 1 - velocity_mae/v_bar) (eval_bc_generalist.
    dynamics_score's formula) -- couples onset detection and velocity accuracy
    multiplicatively, unlike eval/expressive_f1 (harmonic mean of onset_f1 and a
    *matched-only* velocity accuracy, which doesn't penalize misses the way rwmae or
    this does). Computed post-hoc here since it isn't logged live during RL
    training (only eval_bc_generalist.py/eval_fmt_generalist.py's rollout evals
    compute it, and only when a VBAR entry exists for the song)."""
    vbar = df["song"].map(VBAR)
    df["eval/dynamics_score"] = df["eval/onset_f1"] * (
        1.0 - df["eval/velocity_mae"] / vbar
    ).clip(lower=0.0)
    return df


def pull(runs, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_rows, summary_rows = [], []
    for i, r in enumerate(runs, 1):
        c, s = r.config, r.summary
        meta = {
            "run_name": r.name,
            "wandb_id": r.id,
            "stage": stage,
            "song": song_of(c.get("environment_name")),
            "vcoef": c.get("velocity_reward_coef"),
            "ocoef": c.get("onset_accuracy_reward_coef"),
            "seed": c.get("seed"),
            "max_steps": c.get("max_steps"),
        }
        # r.history(keys=METRICS) is a key-INTERSECTION scan (wandb's scan_history):
        # exactly the sparse eval-only rows we want (~120-500/run), NOT the dense
        # every-train-step log that keys-less history() returns. But if a run is
        # missing even one requested key entirely (true for the older base-*
        # baselines, which predate some eval/* metrics), the intersection is empty
        # and it silently drops every column instead of erroring -- so first narrow
        # METRICS to keys this particular run actually has (from its summary, which
        # includes every key ever logged), then scan just those.
        # Only request keys this run's summary actually has -- including a key that's
        # NEVER logged (not just sometimes-missing) makes the whole intersection
        # empty, not just narrower (scan_history requires every requested key
        # present on a row). Fetched as ONE call so the step column and the metric
        # columns are guaranteed row-aligned (they're one intersection, not two
        # separately-scanned frames of possibly different length).
        present = [m for m in METRICS if m != "global_step" and m in s]
        step_key = "global_step" if "global_step" in s else "_step"  # older runs
        # (the base-* baselines) never logged global_step explicitly at all.
        h = r.history(keys=present + [step_key], pandas=True) if present else pd.DataFrame()
        if step_key != "global_step":
            h["global_step"] = h[step_key]
        h = h.reindex(columns=METRICS)  # add back any still-missing metric as NaN
        h = h.dropna(subset=["global_step"])
        for col, val in meta.items():
            h[col] = val
        hist_rows.append(h)

        summary_rows.append({**meta, "state": r.state, **{m: s.get(m) for m in METRICS}})
        print(f"  [{i}/{len(runs)}] {r.name} ({stage}) -- {len(h)} history rows", flush=True)

    hist_df = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if not hist_df.empty:
        hist_df = add_dynamics_score(hist_df)
    if not summary_df.empty:
        summary_df = add_dynamics_score(summary_df)
    return hist_df, summary_df


def main():
    api = wandb.Api()

    print("Stage A (screening, 3M steps)...")
    stageA = list(api.runs(PROJECT, filters={"tags": "reward_coef_sweep_stageA"}))
    hA, sA = pull(stageA, "A")
    hA.to_csv("stageA_history.csv", index=False)
    sA.to_csv("stageA_summary.csv", index=False)
    print(f"Stage A: {len(stageA)} runs, {len(hA)} history rows written.\n")

    print("Stage B (confirmation, 5M steps)...")
    stageB_all = list(api.runs(PROJECT, filters={"tags": "reward_coef_sweep_stageB"}))
    stageB = [r for r in stageB_all if r.state == "finished"]
    skipped = len(stageB_all) - len(stageB)
    if skipped:
        print(f"  skipping {skipped} non-finished Stage B runs (crashed duplicates from "
              f"the 8/28 double-launch incident, see [[cantabile-reward-sweep-plan]])")
    hB, sB = pull(stageB, "B")
    hB.to_csv("stageB_history.csv", index=False)
    sB.to_csv("stageB_summary.csv", index=False)
    print(f"Stage B: {len(stageB)} runs, {len(hB)} history rows written.\n")

    print("Baseline (v=0/o=0, 5M steps, nocturne+clair)...")
    all_runs = list(api.runs(PROJECT))
    baseline = [
        r for r in all_runs
        if r.config.get("velocity_reward_coef") == 0.0
        and r.config.get("onset_accuracy_reward_coef") == 0.0
        and r.config.get("environment_name") in SONG_BY_ENV
        and r.state == "finished"
        and "base" in r.tags
    ]
    hBase, sBase = pull(baseline, "baseline")
    hBase.to_csv("baseline_history.csv", index=False)
    sBase.to_csv("baseline_summary.csv", index=False)
    print(f"Baseline: {len(baseline)} runs, {len(hBase)} history rows written.\n")

    print("Done.")


if __name__ == "__main__":
    main()
