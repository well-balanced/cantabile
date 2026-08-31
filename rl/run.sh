#!/bin/bash
# Unified experiment launcher.
#
# Usage:
#   bash run.sh --method <method> --song <song> [options]
#
# Methods:  base | vel | base_res | vel_res | abl_dof
# Songs:    twinkle | clair | nocturne | gymnopedie | forelise | prelude | waltz | berceuse |
#           reverie | frenchminuet | (etude-12) fsuite1a | fsuite5s | sonatad845 | partita26 |
#           etudewaltz | bagatelle | kreisleriana | fsuite5g | sonata23_2 | golliwogg |
#           sonata2_1 | sonatak279
# DOF:      fingers_only (default) | fingers_wrist | all  [only for abl_dof]
#
# Hyperparameter overrides (method defaults apply if omitted):
#   --vel <float>        velocity_reward_coef
#   --onset <float>      onset_accuracy_reward_coef
#   --alpha <float>      residual_alpha
#   --max-steps <int>    total env steps (default 5000000 -- NOTE: base_residual/
#                        vel_res load checkpoint_5000000.flax by name, so
#                        overriding this for a *base-checkpoint-producing* run breaks
#                        that reference. Safe to override for standalone baselines
#                        (e.g. the 8M-step protocol for pure `base`/`vel` runs
#                        not meant to be reused as a residual base).
#
# Examples:
#   bash run.sh --method base --song twinkle --seed 0 --gpu 0
#   bash run.sh --method vel --song clair --vel 0.3 --onset 0.4
#   bash run.sh --method abl_dof --song twinkle --gpu 2 --dof fingers_wrist

set -euo pipefail

GPU=0
SEED=0
DOF="fingers_only"
METHOD=""
SONG=""
SUFFIX=""
DRY_RUN=false
# Override sentinels (empty = use method default)
OVR_VEL=""
OVR_ONSET=""
OVR_ALPHA=""
OVR_BASE_CKPT=""
MAX_STEPS=5000000
STYLE_SCALE=1.0
STYLE_BIAS=0.0
STYLE_CONTRAST=1.0
STYLE_TREND=0.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)     METHOD="$2";       shift 2 ;;
        --song)       SONG="$2";         shift 2 ;;
        --seed)       SEED="$2";         shift 2 ;;
        --gpu)        GPU="$2";          shift 2 ;;
        --dof)        DOF="$2";          shift 2 ;;
        --suffix)     SUFFIX="$2";       shift 2 ;;
        --vel)        OVR_VEL="$2";      shift 2 ;;
        --onset)      OVR_ONSET="$2";    shift 2 ;;
        --alpha)      OVR_ALPHA="$2";    shift 2 ;;
        --base-ckpt)  OVR_BASE_CKPT="$2"; shift 2 ;;
        --max-steps)  MAX_STEPS="$2";     shift 2 ;;
        --scale)      STYLE_SCALE="$2";  shift 2 ;;
        --bias)       STYLE_BIAS="$2";   shift 2 ;;
        --contrast)   STYLE_CONTRAST="$2"; shift 2 ;;
        --trend)      STYLE_TREND="$2";  shift 2 ;;
        --dry-run)    DRY_RUN=true;      shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$METHOD" ]] && { echo "Error: --method is required"; exit 1; }
[[ -z "$SONG" ]]   && { echo "Error: --song is required";   exit 1; }

# Song → environment name
case $SONG in
    twinkle)    ENV="RoboPianist-debug-TwinkleTwinkleRousseau-v0" ;;
    clair)      ENV="RoboPianist-debug-ClairDeLune-v0" ;;
    nocturne)   ENV="RoboPianist-debug-NocturneRousseau-v0" ;;
    gymnopedie) ENV="RoboPianist-debug-GymnopedieNo1-v0" ;;
    forelise)   ENV="RoboPianist-debug-ForElise-v0" ;;
    prelude)    ENV="RoboPianist-debug-PreludeOp28No7-v0" ;;
    waltz)      ENV="RoboPianist-debug-WaltzOp64No1-v0" ;;
    berceuse)      ENV="RoboPianist-debug-Berceuse-v0" ;;
    reverie)       ENV="RoboPianist-debug-Reverie-v0" ;;
    frenchminuet)  ENV="RoboPianist-debug-FrenchSuiteNo3Minuet-v0" ;;
    # RoboPianist-etude-12 (original RoboPianist paper's standard 12-etude benchmark).
    fsuite1a)   ENV="RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0" ;;
    fsuite5s)   ENV="RoboPianist-etude-12-FrenchSuiteNo5Sarabande-v0" ;;
    sonatad845) ENV="RoboPianist-etude-12-PianoSonataD8451StMov-v0" ;;
    partita26)  ENV="RoboPianist-etude-12-PartitaNo26-v0" ;;
    etudewaltz) ENV="RoboPianist-etude-12-WaltzOp64No1-v0" ;;
    bagatelle)  ENV="RoboPianist-etude-12-BagatelleOp3No4-v0" ;;
    kreisleriana) ENV="RoboPianist-etude-12-KreislerianaOp16No8-v0" ;;
    fsuite5g)   ENV="RoboPianist-etude-12-FrenchSuiteNo5Gavotte-v0" ;;
    sonata23_2) ENV="RoboPianist-etude-12-PianoSonataNo232NdMov-v0" ;;
    golliwogg)  ENV="RoboPianist-etude-12-GolliwoggsCakewalk-v0" ;;
    sonata2_1)  ENV="RoboPianist-etude-12-PianoSonataNo21StMov-v0" ;;
    sonatak279) ENV="RoboPianist-etude-12-PianoSonataK279InCMajor1StMov-v0" ;;
    *) echo "Unknown song: $SONG"; exit 1 ;;
esac

# Method → reward coefficients, residual args, run name
RESIDUAL_ARGS=""

case $METHOD in
    base)
        VEL=0.0
        ONSET=0.0
        RUN_NAME="base-${SONG}-s${SEED}"
        GROUP="base"
        ;;
    vel)
        VEL=0.2
        ONSET=0.5
        RUN_NAME="vel-${SONG}-s${SEED}"
        GROUP="vel"
        ;;
    base_res)
        VEL=0.5
        ONSET=0.5
        BASE_CKPT="./checkpoints/base/${SONG}/seed${SEED}/checkpoint_5000000.flax"
        RESIDUAL_ARGS="--residual-alpha 0.2 --residual-action-mode fingers_only --base-checkpoint ${BASE_CKPT}"
        RUN_NAME="base_res-${SONG}-s${SEED}"
        GROUP="base_res"
        ;;
    vel_res)
        VEL=0.5
        ONSET=0.5
        BASE_CKPT="./checkpoints/vel/${SONG}/seed${SEED}/checkpoint_5000000.flax"
        RESIDUAL_ARGS="--residual-alpha 0.2 --residual-action-mode fingers_only --base-checkpoint ${BASE_CKPT}"
        RUN_NAME="vel_res-${SONG}-s${SEED}"
        GROUP="vel_res"
        ;;
    abl_dof)
        VEL=0.5
        ONSET=0.5
        BASE_CKPT="./checkpoints/base/${SONG}/seed${SEED}/checkpoint_5000000.flax"
        case $DOF in
            fingers_only)  DOF_TAG="fingers" ;;
            fingers_wrist) DOF_TAG="wrist"   ;;
            all)           DOF_TAG="full"     ;;
            *) echo "Unknown dof: $DOF (use fingers_only | fingers_wrist | all)"; exit 1 ;;
        esac
        RESIDUAL_ARGS="--residual-alpha 0.2 --residual-action-mode ${DOF} --base-checkpoint ${BASE_CKPT}"
        RUN_NAME="abl_dof-${DOF_TAG}-${SONG}-s${SEED}"
        GROUP="abl_dof"
        ;;
    style_specialist)
        VEL=0.5
        ONSET=0.5
        BASE_CKPT="./checkpoints/vel/${SONG}/seed${SEED}/checkpoint_5000000.flax"
        RESIDUAL_ARGS="--residual-alpha 0.2 --residual-action-mode fingers_only --base-checkpoint ${BASE_CKPT}"
        RUN_NAME="style_specialist-${SONG}-s${SEED}"
        GROUP="style_specialist"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Valid methods: base | vel | base_res | vel_res | abl_dof | style_specialist"
        exit 1
        ;;
esac

# Apply hyperparameter overrides.
[[ -n "$OVR_VEL" ]]   && VEL="$OVR_VEL"
[[ -n "$OVR_ONSET" ]] && ONSET="$OVR_ONSET"
if [[ -n "$OVR_ALPHA" && -n "$RESIDUAL_ARGS" ]]; then
    RESIDUAL_ARGS=$(echo "$RESIDUAL_ARGS" | sed "s/--residual-alpha [^ ]*/--residual-alpha ${OVR_ALPHA}/")
fi
if [[ -n "$OVR_BASE_CKPT" && -n "$RESIDUAL_ARGS" ]]; then
    RESIDUAL_ARGS=$(echo "$RESIDUAL_ARGS" | sed "s|--base-checkpoint [^ ]*|--base-checkpoint ${OVR_BASE_CKPT}|")
fi

[[ -n "$SUFFIX" ]] && RUN_NAME="${RUN_NAME}-${SUFFIX}"

echo "Launching: ${RUN_NAME} on GPU ${GPU}"
echo "  env:     ${ENV}"
echo "  vel:     ${VEL}  onset: ${ONSET}  max-steps: ${MAX_STEPS}"
[[ -n "$RESIDUAL_ARGS" ]] && echo "  residual: ${RESIDUAL_ARGS}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] exiting without launching."
    exit 0
fi

WANDB_DIR=./ \
MUJOCO_GL=egl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=$GPU \
MUJOCO_EGL_DEVICE_ID=$GPU \
../.venv/bin/python train.py \
    --root-dir ./tmp \
    --warmstart-steps 5000 \
    --max-steps $MAX_STEPS \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "$ENV" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 5 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --mode "online" \
    --checkpoint-interval 2000000 \
    --seed $SEED \
    --velocity-reward-coef $VEL \
    --onset-accuracy-reward-coef $ONSET \
    --style-velocity-scale $STYLE_SCALE \
    --style-velocity-bias $STYLE_BIAS \
    --style-velocity-contrast $STYLE_CONTRAST \
    --style-dynamic-trend $STYLE_TREND \
    --name "$RUN_NAME" \
    --tags "$GROUP" \
    $RESIDUAL_ARGS
