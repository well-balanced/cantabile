# Experiment Protocol

## Motivation

Existing robotic piano systems optimize for note accuracy but treat velocity as flat or incidental. The open question is whether this environment — physics simulation + shadow hands — can actually *measure and reproduce* velocity dynamics at all, and if so, whether a policy can be trained to track a target velocity contour without sacrificing note accuracy.

---

## Research Questions

1. Does the environment have sufficient dynamic resolution to measure velocity differences — i.e., can it distinguish flat, soft, GT, and loud velocity contours in robot output?
2. Does adding an onset accuracy reward increase match rate above the base — expanding the set of notes on which velocity can be optimized?
3. Given higher match rate, does velocity reward improve velocity accuracy on those matched notes?
4. Does the residual approach preserve onset F1 and match rate while achieving (3) — validating that velocity learning does not crowd out accuracy?

> **Why match rate is the key prerequisite:** velocity MAE and correlation are computed only over matched onsets. If match rate is low, even perfect velocity on matched notes produces negligible perceptual improvement. Onset reward must first expand the "optimizable pie" before velocity reward can have audible effect.

---

## Songs

| ID | Environment Name |
|----|-----------------|
| `twinkle` | `RoboPianist-debug-TwinkleTwinkleRousseau-v0` |
| `clair` | `RoboPianist-debug-ClairDeLuneRousseau-v0` |
| `nocturne` | `RoboPianist-debug-NocturneRousseau-v0` |

---

## Fixed Hyperparameters

```yaml
control_timestep: 0.05
n_steps_lookahead: 10
trim_silence: true
gravity_compensation: true
reduced_action_space: true

hidden_dims: [256, 256, 256]
critic_dropout_rate: 0.01
critic_layer_norm: true
discount: 0.8
warmstart_steps: 5000

residual_alpha: 0.2
residual_action_mode: fingers_only
freeze_base: true

max_steps: 5_000_000
eval_interval: 10_000
eval_episodes: 5
final_eval_episodes: 20
final_eval_deterministic: true
```

Onset accuracy sub-weights (all methods that use it): `hit=1.0`.

---

## W&B Conventions

```
project: robopianist_dynamics
group:   "{group_id}"
name:    "{group_id}-{song}-s{seed}"
```

---

## Experiment Groups

---

### Group 1 — Contour Validation

**Rationale:** Before any training, verify that the environment can actually measure velocity differences. Velocity MAE and correlation are only meaningful if the environment has sufficient dynamic resolution to distinguish different playing intensities. This is RQ1, and it is a prerequisite for the rest of the paper.

**Verifies:** RQ1

**Protocol:** Evaluate a trained base policy under four fixed velocity conditions by overriding the MIDI velocity at eval time. No additional training.

| Condition | Description |
|-----------|-------------|
| `flat_vel` | All notes at a fixed MIDI velocity (environment default) |
| `gt_vel` | Original score velocity contour |
| `soft_vel` | GT contour scaled down uniformly (e.g. × 0.5) |
| `loud_vel` | GT contour scaled up uniformly (e.g. × 1.5, clipped at 127) |

**W&B runs:**

| Group | Run name | Song |
|-------|----------|------|
| `contour_val` | `contour_val-twinkle-flat` | twinkle |
| `contour_val` | `contour_val-twinkle-gt` | twinkle |
| `contour_val` | `contour_val-twinkle-soft` | twinkle |
| `contour_val` | `contour_val-twinkle-loud` | twinkle |

**Expected result:** Four clearly separated velocity contour curves. If contours overlap, the environment lacks dynamic resolution and the velocity reward has no meaningful signal.

---

### Group 2 — Base

**Rationale:** Establish the note accuracy ceiling before any velocity objective is introduced. This policy is also used as the frozen base for base_residual. Without a strong base, the base_residual ablation is confounded — a weak base forces the residual to compensate for poor note accuracy rather than adding velocity control.

**Verifies:** Reference for RQ2, RQ3, RQ4.

**Reward:** `velocity_reward_coef=0.0`, `onset_accuracy_reward_coef=0.0`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `base` | `base-twinkle-s0` |
| `base` | `base-clair-s0` |
| `base` | `base-nocturne-s0` |

**Expected result:** High onset F1 and key F1. Velocity contour is flat or random — no dynamics tracking. Match rate is the reference floor that all subsequent methods must exceed.

---

### Group 3 — Vel-Aware (end-to-end baseline)

**Rationale:** Train the full policy end-to-end with both velocity and onset accuracy rewards. This is the simplest approach to adding velocity awareness and serves as the base policy for the main method (vel_aware_residual). It also reveals the core tension: velocity reward competes with note accuracy when optimized jointly, causing F1 to drop relative to base.

**Verifies:** RQ2, RQ3. Establishes the tension between velocity learning and note accuracy (motivates residual approach). Also provides the frozen base for Group 5.

**Reward:** `velocity_reward_coef=0.2`, `onset_accuracy_reward_coef=0.5`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `vel_aware` | `vel_aware-twinkle-s0` |
| `vel_aware` | `vel_aware-clair-s0` |
| `vel_aware` | `vel_aware-nocturne-s0` |

**Expected result:**
- Match rate and onset F1 > base (onset reward active)
- Velocity MAE improved over base
- F1 slightly lower than base — velocity reward crowds out accuracy; this tension motivates the residual approach

---

### Group 4 — Base + Residual *(ablation)*

**Rationale:** Freeze the base policy (F1-only, no velocity awareness) and train a residual actor with velocity and onset rewards. This ablation isolates the contribution of the residual architecture from the contribution of the vel_aware base. Comparing base_residual vs vel_aware_residual answers: how much does starting from a velocity-aware base matter? Comparing base_residual vs vel_aware answers: is residual on a naive base better or worse than end-to-end velocity training?

**Verifies:** Partial RQ4. Isolates the residual architecture's contribution independent of base quality.

**Reward:** `velocity_reward_coef=0.5`, `onset_accuracy_reward_coef=0.5`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `base_residual` | `base_residual-twinkle-s0` |
| `base_residual` | `base_residual-clair-s0` |
| `base_residual` | `base_residual-nocturne-s0` |

**Expected result:**
- F1 ≈ base (frozen base + small α preserves note accuracy)
- Velocity MAE better than base, but worse than vel_aware_residual (base has no prior velocity knowledge)
- Together with vel_aware_residual, shows that vel_aware base provides meaningful signal for the residual to build on

---

### Group 5 — Vel-Aware + Residual *(main method)*

**Rationale:** Freeze the vel_aware policy and train a residual actor on top. The vel_aware base already understands velocity contours to some degree; the residual only needs to refine. This two-stage approach achieves the best of both worlds: the vel_aware base provides velocity-relevant representations, while the frozen + residual structure prevents the velocity objective from crowding out note accuracy. Expected to be the best method on velocity metrics while maintaining F1 at or near base level.

**Verifies:** RQ3, RQ4. Main contribution.

**Reward:** `velocity_reward_coef=0.5`, `onset_accuracy_reward_coef=0.5`
**Base:** `checkpoints/vel_aware/{song}/seed0`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `vel_aware_residual` | `vel_aware_residual-twinkle-s0` |
| `vel_aware_residual` | `vel_aware_residual-clair-s0` |
| `vel_aware_residual` | `vel_aware_residual-nocturne-s0` |

**Expected result:**
- Velocity MAE lowest across all methods
- Velocity correlation highest across all methods
- F1 ≈ base (frozen base + small α), similar to base_residual
- This validates the two-stage design: vel_aware pre-training provides velocity-relevant representations, residual fine-tuning refines without sacrificing accuracy

---

### Group 6 — Residual Action Mode Ablation

**Rationale:** The main experiment fixes `residual_action_mode=fingers_only`, which restricts the residual to finger joints only (wrist, forearm, sustain unchanged). This choice is motivated by the hypothesis that velocity is primarily controlled through finger keystroke dynamics, not wrist posture. This ablation tests whether relaxing this constraint to include wrist DOFs or all DOFs changes the result. Run on twinkle only for fast iteration.

**Verifies:** Whether the `fingers_only` constraint is necessary for F1 preservation, or just a conservative default.

**Reward:** Same as Group 5. **Base:** `checkpoints/base/twinkle/seed0`

**W&B runs:**

| Group | Run name | DOFs |
|-------|----------|------|
| `abl_dof` | `abl_dof-fingers-twinkle-s0` | fingers only (same as Group 5 — reference) |
| `abl_dof` | `abl_dof-wrist-twinkle-s0` | fingers + wrist |
| `abl_dof` | `abl_dof-full-twinkle-s0` | all DOFs |

**Expected result:**
- `fingers` ≈ `wrist` for vel MAE (wrist doesn't add much to keystroke intensity)
- `full` may show slightly better vel MAE but onset F1 degradation (residual can now move the hands away from correct keys)
- If `wrist` is strictly better with no F1 cost, update main experiment default

---

## Paper Main Table

| Method | Match Rate ↑ | Onset F1 ↑ | Key F1 ↑ | Vel MAE ↓ | Vel Corr ↑ |
|--------|-------------|-----------|---------|----------|-----------|
| base | ref | ref | ref | — | — |
| vel_aware | | | | | |
| base_residual (ablation) | | | | | |
| **vel_aware_residual (ours)** | | | | | |

Key comparisons:
- **vel_aware vs base**: does velocity reward help? (RQ3)
- **vel_aware vs vel_aware_residual**: does residual fine-tuning improve over end-to-end? (RQ4)
- **base_residual vs vel_aware_residual**: how much does the vel_aware base contribute? (attribution)
- **base_residual vs vel_aware**: is residual on naive base competitive with end-to-end?

---

## Checkpoint Layout

```
checkpoints/
  base/
    twinkle/seed0/
    clair/seed0/
    nocturne/seed0/
  vel_aware/
    twinkle/seed0/
    clair/seed0/
    nocturne/seed0/
  residual/
    base_residual/twinkle/seed0/
    base_residual/clair/seed0/
    base_residual/nocturne/seed0/
    vel_aware_residual/twinkle/seed0/
    vel_aware_residual/clair/seed0/
    vel_aware_residual/nocturne/seed0/
  ablation/
    dof_fingers/twinkle/seed0/
    dof_wrist/twinkle/seed0/
    dof_full/twinkle/seed0/
```

---

## Metrics (log every eval)

```yaml
pie:
  - match_rate              # prerequisite: must increase with onset reward

accuracy:                   # guardrail: must not drop from base
  - onset_f1
  - key_f1

dynamics:                   # meaningful only when match_rate is high
  - velocity_mae
  - velocity_corr
  - velocity_rmse
  - velocity_bias

diagnostics:
  - mean_robot_velocity
  - mean_gt_velocity
  - num_matched_onsets
```

---

## All Groups & Runs

- **G1 — Contour Validation**
  - `contour_val-twinkle-flat`
  - `contour_val-twinkle-gt`
  - `contour_val-twinkle-soft`
  - `contour_val-twinkle-loud`

- **G2 — Base**
  - `base-twinkle-s0`
  - `base-clair-s0`
  - `base-nocturne-s0`

- **G3 — Vel-Aware** *(end-to-end baseline, also base for main method)*
  - `vel_aware-twinkle-s0`
  - `vel_aware-clair-s0`
  - `vel_aware-nocturne-s0`

- **G4 — Base + Residual** *(ablation)*
  - `base_residual-twinkle-s0`
  - `base_residual-clair-s0`
  - `base_residual-nocturne-s0`

- **G5 — Vel-Aware + Residual** *(main method)*
  - `vel_aware_residual-twinkle-s0`
  - `vel_aware_residual-clair-s0`
  - `vel_aware_residual-nocturne-s0`

- **G6 — DOF Ablation**
  - `abl_dof-fingers-twinkle-s0`
  - `abl_dof-wrist-twinkle-s0`
  - `abl_dof-full-twinkle-s0`

---

## Run Priority

```
Phase 1:  base → vel_aware          3 + 3 runs
Gate:     vel_aware shows velocity improvement with some F1 cost vs base

Phase 2:  vel_aware_residual        3 runs   (main method, needs vel_aware checkpoint)
Gate:     vel MAE lowest, F1 ≈ base

Phase 3:  base_residual             3 runs   (ablation, needs base checkpoint)
Gate:     F1 preserved, vel MAE between base and vel_aware_residual

Phase 4:  DOF ablation              3 runs, twinkle only

Phase 5:  Paper-grade, seeds 0/1/2 on all methods    36 runs total
```
