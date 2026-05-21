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

Onset accuracy sub-weights (all methods that use it): `hit=1.0, hold_rehit=0.5`.

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

**Rationale:** Establish the note accuracy ceiling before any velocity objective is introduced. This policy is also used as the frozen base for Group 3 and Group 5. Without a strong base, residual experiments are confounded — a weak base means residual must compensate for poor note accuracy, not just add velocity control.

**Verifies:** Reference for RQ2, RQ3, RQ4. Establishes that note accuracy is achievable before adding velocity complexity.

**Reward:** `velocity_reward_coef=0.0`, `onset_accuracy_reward_coef=0.0`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `base` | `base-twinkle-s0` |
| `base` | `base-clair-s0` |
| `base` | `base-nocturne-s0` |

**Expected result:** High onset F1 and key F1. Velocity contour is flat or random — no dynamics tracking. Match rate is the reference floor that all subsequent methods must exceed.

---

### Group 3 — Base + Residual *(main method)*

**Rationale:** Core contribution. Freeze the base policy and train a small residual actor with velocity and onset rewards. The residual can only modulate finger actions by ±α — it cannot undo what the base does. This structural constraint is why F1 should be preserved: the base's note-hitting behavior is frozen, and the residual only fine-tunes keystroke intensity. The onset accuracy reward additionally pushes match rate above the base, expanding the set of notes available for velocity optimization.

**Verifies:** RQ2 (match rate ↑ via onset reward), RQ3 (vel MAE ↓ via velocity reward), RQ4 (F1 preserved by frozen base + structural constraint).

**Reward:** `velocity_reward_coef=0.5`, `onset_accuracy_reward_coef=0.5`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `base_residual` | `base_residual-twinkle-s0` |
| `base_residual` | `base_residual-clair-s0` |
| `base_residual` | `base_residual-nocturne-s0` |

**Expected result:**
- Match rate > Group 2 (onset reward expands the pie)
- Onset F1 ≥ Group 2, ideally higher (onset reward actively improves accuracy)
- Vel MAE lower than Group 2 (velocity reward effective on expanded match set)
- Key F1 ≈ Group 2 (frozen base + small α preserves non-onset behavior)

---

### Group 4 — Vel-Aware (end-to-end baseline)

**Rationale:** Compare against a simpler baseline: train the full policy end-to-end with both velocity and onset accuracy rewards, without freezing any part of the network. This answers whether the residual structure is necessary, or whether just adding the right rewards to a full policy achieves the same result. Expected to be less stable than Group 3 because the policy must simultaneously learn note accuracy and velocity dynamics from scratch without the structural protection of a frozen base.

**Verifies:** RQ3 and RQ4 from a different angle. If Group 4 performs similarly to Group 3, the residual structure adds little. If Group 4 shows higher F1 variance or velocity reward crowding out accuracy, it validates the residual design.

**Reward:** `velocity_reward_coef=0.2`, `onset_accuracy_reward_coef=0.5`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `vel_aware` | `vel_aware-twinkle-s0` |
| `vel_aware` | `vel_aware-clair-s0` |
| `vel_aware` | `vel_aware-nocturne-s0` |

**Expected result:**
- Match rate and onset F1 > Group 2 (onset reward active)
- Vel MAE improved but training less stable than Group 3
- F1 may show higher variance across songs; risk of velocity reward crowding out accuracy on harder songs

---

### Group 5 — Vel-Aware + Residual

**Rationale:** Test whether the residual approach generalizes — does it add further improvement even when the base already has velocity awareness? If yes, the residual is not just a shortcut for a weak base but a generally useful fine-tuning mechanism. If no, the residual's benefit is specific to F1-only bases and the structural constraint is doing the work.

**Verifies:** Generalizability of residual (extends RQ4). Provides the Group 4 vs Group 5 comparison.

**Reward:** `velocity_reward_coef=0.5`, `onset_accuracy_reward_coef=0.5`
**Base:** `checkpoints/vel_aware/{song}/seed0`

**W&B runs:**

| Group | Run name |
|-------|----------|
| `vel_aware_residual` | `vel_aware_residual-twinkle-s0` |
| `vel_aware_residual` | `vel_aware_residual-clair-s0` |
| `vel_aware_residual` | `vel_aware_residual-nocturne-s0` |

**Expected result:**
- Vel MAE further improved vs Group 4
- Match rate and onset F1 maintained or slightly improved
- If this group does not outperform Group 4, residual on vel_aware base provides no benefit — still informative as a negative result

---

### Group 6 — Residual Action Mode Ablation

**Rationale:** The main experiment fixes `residual_action_mode=fingers_only`, which restricts the residual to finger joints only (wrist, forearm, sustain unchanged). This choice is motivated by the hypothesis that velocity is primarily controlled through finger keystroke dynamics, not wrist posture. This ablation tests whether relaxing this constraint to include wrist DOFs or all DOFs changes the result. Run on twinkle only for fast iteration.

**Verifies:** Whether the `fingers_only` constraint is necessary for F1 preservation, or just a conservative default.

**Reward:** Same as Group 3. **Base:** `checkpoints/base/twinkle/seed0`

**W&B runs:**

| Group | Run name | DOFs |
|-------|----------|------|
| `abl_dof` | `abl_dof-fingers-twinkle-s0` | fingers only (same as Group 3 — reference) |
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
| base (G2) | ref | ref | ref | — | — |
| base_residual (G3) | | | | | |
| vel_aware (G4) | | | | | |
| vel_aware_residual (G5) | | | | | |

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
  - onset_hold_rehit_rate
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

- **G3 — Base + Residual**
  - `base_residual-twinkle-s0`
  - `base_residual-clair-s0`
  - `base_residual-nocturne-s0`

- **G4 — Vel-Aware**
  - `vel_aware-twinkle-s0`
  - `vel_aware-clair-s0`
  - `vel_aware-nocturne-s0`

- **G5 — Vel-Aware + Residual**
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
Phase 1:  G2 → G3        3 + 3 runs
Gate:     match_rate ↑, onset_f1 ↑, vel_mae ↓ vs G2

Phase 2:  G4 → G5        3 + 3 runs
Gate:     compare G3 vs G4 stability

Phase 3:  G6             3 runs, twinkle only

Phase 4:  Paper-grade, seeds 0/1/2 on G2–G5      36 runs total
```
