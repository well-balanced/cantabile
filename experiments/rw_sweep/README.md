# reward-coef sweep dataset

Full wandb history (not just endpoint values) for the Phase 2 velocity/onset
reward-coefficient sweep, pulled for paper figures that need metrics-over-training-
steps, not just final numbers. Regenerate any time with:

```
../../.venv/bin/python build_dataset.py
```

It always re-pulls from the wandb API (`cantabile/cantabile` project) -- no local
caching, safe to re-run.

## Files

Each stage has a `*_history.csv` (one row per run per eval-logging step -- the
time-series data for figures) and a `*_summary.csv` (one row per run, final/latest
values only -- for quick lookups without loading the full history).

- **`stageA_history.csv` / `stageA_summary.csv`** -- screening sweep. 25
  `velocity_reward_coef` x `onset_accuracy_reward_coef` combos in
  `{0.1,0.2,0.3,0.5,1.0}^2`, x 2 songs (nocturne, clair), seed 43 only, 3M steps.
  50 runs.
- **`stageB_history.csv` / `stageB_summary.csv`** -- confirmation run. 8 candidates
  picked from Stage A x 2 songs x 3 seeds (43/44/45), 5M steps. 48 runs (finished
  only -- see caveat below).
- **`baseline_history.csv` / `baseline_summary.csv`** -- `velocity_reward_coef=0,
  onset_accuracy_reward_coef=0` (no velocity/onset reward at all), nocturne+clair,
  5M steps, 8 runs (seeds 0/42/43/44 depending on song -- these are pre-existing
  `base-*` runs, not launched as part of this sweep). Used as the "does frame-f1
  drop vs. not having the reward at all" guardrail check.

## Columns

`run_name, wandb_id, stage, song, vcoef, ocoef, seed, max_steps` identify the run.
`global_step` is the training step for that row (`_history.csv` only).

Metric columns, all under the `eval/` prefix wandb logs them with:

- **`eval/f1`, `eval/precision`, `eval/recall`** -- **frame-level** key-press F1
  (sklearn `precision_recall_fscore_support` per timestep,
  `evaluation.py`'s `get_musical_metrics()`). Runs high (0.7-0.9) and barely moves
  across reward-coef combos -- most timesteps correctly have no key pressed. Used
  as a **guardrail** (did this config break basic key-pressing?), not a ranking
  metric.
- **`eval/onset_f1`, `eval/onset_precision`, `eval/onset_recall`,
  `eval/onset_hit_rate`, `eval/onset_fp_rate`, `eval/onset_miss_rate`,
  `eval/match_rate`** -- **event-level** onset-detection metrics
  (`get_velocity_metrics()`). Runs low (0.15-0.55), far more discriminating.
  **`eval/f1` and `eval/onset_f1` are different metrics that happen to share the
  `eval/` prefix -- do not conflate them** (Pearson r ~0.48 across Stage A, i.e.
  they genuinely diverge).
- **`eval/velocity_mae`** -- MAE over matched onsets only (excludes misses,
  optimistic).
- **`eval/velocity_mae_recall_weighted`** ("rwmae") -- `recall * matched_mae + (1 -
  recall) * v_bar`; misses charged at v_bar (the song's mean onset velocity) instead
  of excluded. The primary velocity-accuracy ranking metric used for this sweep.
  Lower is better.
- **`eval/velocity_correlation`** ("corr") -- Pearson correlation between robot and
  ground-truth onset velocities. Secondary metric; the one that most sharply
  penalizes "hits the onset, ignores dynamics" configs (e.g. high-`ocoef` combos).
- **`eval/velocity_rmse`, `eval/velocity_std`, `eval/velocity_bias`** -- other
  velocity-error decompositions, not used in the ranking so far but kept for
  completeness.
- **`eval/expressive_f1`** -- harmonic mean of onset_f1 and velocity accuracy.
- **`eval/sustain_f1`, `eval/sustain_precision`, `eval/sustain_recall`** -- pedal
  sustain accuracy, not used in this sweep's ranking.
- **`eval/return`, `eval/length`** -- episode return / length, for sanity checks.

- **`eval/dynamics_score`** -- `onset_f1 * max(0, 1 - velocity_mae/v_bar)`
  (`eval_bc_generalist.py`'s `dynamics_score()`). Not logged live during RL
  training (only the rollout-eval scripts compute it, and only when a `VBAR` entry
  exists for the song) -- computed here post-hoc by `build_dataset.py` using
  `v_bar = {"nocturne": 39.6068, "clair": 44.1646}`, reconstructed from this sweep's
  own `eval/velocity_mae_recall_weighted` (see `VBAR` comment in the code below).
  **Prefer this over `eval/expressive_f1`** as the combined onset+velocity metric:
  expressive_f1 is a harmonic mean of onset_f1 and a *matched-onset-only* velocity
  accuracy (so it doesn't charge anything for misses the way rwmae/dynamics_score
  do), where dynamics_score's multiplicative form more directly answers "does this
  config get both the timing and the dynamics right." Caveat: dynamics_score uses
  `velocity_mae` (matched-only), not `velocity_correlation` -- it does NOT catch the
  "hits onsets but with near-random/negative dynamics correlation" failure mode
  `velocity_correlation` catches (see `v0.3-o1.0` in the table below: 2nd-best
  dynamics_score despite a negative clair correlation). Still check corr separately.

## Known caveats

- **Stage B has 11 excluded `crashed` runs**: on 2026-08-28 a duplicate driver was
  accidentally launched on top of the real one (same 48-job queue twice) and the
  duplicate's 11 already-started jobs were `kill -9`'d once caught. Their wandb runs
  show up as `crashed` and are excluded by `build_dataset.py` (filters to
  `state == "finished"`) -- the real (surviving) run for each of those 11 configs is
  included normally. See `[[cantabile-reward-sweep-plan]]` memory for detail.
- **Baseline seeds don't match Stage A/B's (43/44/45)**: the `base-*` runs predate
  this sweep and use whatever seeds they were originally launched with (0/42/43/44).
  Only 43/44 overlap by number; treat the baseline comparison as a 3-seed-mean vs.
  3-seed-mean comparison, not a seed-matched paired one.
- **Baseline runs predate some metrics**: the older `base-*` runs don't have
  `eval/onset_precision`, `eval/onset_recall`, or `eval/velocity_mae_recall_weighted`
  logged at all (those were added mid-project, see `[[cantabile-reward-sweep-plan]]`)
  -- those columns are genuinely `NaN` for baseline rows, not a pull bug.

## Stage B final results (as of 2026-08-30)

8 candidates x 2 songs x 3 seeds (43/44/45), 5M steps, 48/48 finished. Friedman
test across the 8 candidates (block = song x seed, n=6): **rwmae not significant**
(chi2=9.00, p=0.253), **corr not significant** (chi2=11.11, p=0.134), **onset_f1
significant** (chi2=27.51, **p=0.00027**), f1 not significant (chi2=8.83, p=0.265).
Practical read: at this point in the sweep the 8 finalists are statistically tied
on velocity accuracy -- onset_f1 is the only axis that reliably separates them.

Per-candidate means (6 blocks each), sorted by rwmae (primary metric, lower=better):

| v | o | rwmae↓ | corr↑ | onset_f1↑ | f1 | dyn_score↑ | noc f1 vs base | clair f1 vs base |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.5 | 27.07 | 0.409 | 0.514 | 0.836 | **0.359** | -0.0215 | -0.0303 |
| 0.3 | 1.0 | 27.25 | 0.140 | **0.545** | 0.834 | 0.354 | -0.0433 | -0.0120 |
| 0.5 | 0.5 | 27.42 | **0.522** | 0.439 | 0.830 | 0.341 | -0.0333 | -0.0313 |
| 0.5 | 0.3 | 28.09 | 0.476 | 0.438 | 0.833 | 0.340 | -0.0326 | -0.0259 |
| 0.3 | 0.3 | 28.69 | 0.448 | 0.446 | 0.846 | 0.323 | -0.0353 | +0.0042 |
| 0.2 | 0.5 | 28.82 | 0.307 | 0.500 | 0.857 | 0.315 | **-0.0077** | **-0.0015** |
| 0.2 | 0.2 | 28.96 | 0.401 | 0.479 | **0.861** | 0.319 | -0.0107 | +0.0090 |
| 0.5 | 0.2 | 30.35 | 0.469 | 0.365 | 0.827 | 0.286 | -0.0627 | -0.0078 |

`... vs base` = mean Stage B f1 minus the `v=0/o=0` baseline mean for that song
(baseline: nocturne 0.826±0.012 (n=4), clair 0.898±0.022 (n=4) -- see
`baseline_summary.csv`). **This guardrail is the decisive cut**: every candidate
except v0.2-o0.5 and (marginally) v0.2-o0.2 drops frame-f1 on nocturne well beyond
baseline noise. v0.2-o0.5's drop is small on both songs (-0.008 / -0.002, both
comfortably inside baseline std); v0.2-o0.2's nocturne drop (-0.011) sits just past
a -0.01 tolerance band -- a genuinely borderline call, not a clean pass.

Cross-song robustness (worst of nocturne/clair per candidate) also favors the same
region: `v0.3-o0.5` has the best worst-song rwmae (28.24) and by far the best
worst-song corr (0.301 vs the next-best ~0.22), but it fails the f1-vs-baseline
guardrail on **both** songs -- so it's a strong rwmae/corr candidate that the
guardrail rules out, not a candidate the guardrail confirms.

**Net**: no single candidate wins on every axis. `v0.2-o0.5` is the cleanest
guardrail-pass (doesn't hurt frame-f1 vs. no-reward-at-all, on either song) with
still-respectable onset_f1 (0.500, 3rd of 8) and dynamics_score (0.315), at the cost
of a middling corr (0.307, worst-song 0.108). `v0.3-o0.5` wins rwmae/corr/dyn_score
outright but fails the guardrail. This tension (does the guardrail or the
velocity-accuracy ranking take priority) is not yet resolved -- next step is
deciding that, not re-running more sweeps.
