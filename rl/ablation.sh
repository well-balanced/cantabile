#!/bin/bash

DEVICE=${1:-0}
SONG=${2:-tw}
METHOD=${3:-base}

case $SONG in
    tw) ENV="RoboPianist-debug-TwinkleTwinkleRousseau-v0" ;;
    nt) ENV="RoboPianist-debug-NocturneRousseau-v0" ;;
    *) echo "Usage: bash ablation.sh <gpu> <tw|nt> <base|e2e|residual> [base_ckpt]"; exit 1 ;;
esac

case $METHOD in
    base)
        VEL=0.0
        ONSET=0.0
        EXTRA_ARGS=""
        ;;
    e2e)
        VEL=0.5
        ONSET=0.5
        EXTRA_ARGS=""
        ;;
    residual)
        VEL=0.5
        ONSET=0.5
        BASE_CKPT=${4:?"residual requires base checkpoint as 4th arg"}
        EXTRA_ARGS="--residual-alpha 0.1 --residual-action-mode fingers_only --base-checkpoint $BASE_CKPT"
        ;;
    *) echo "Usage: bash ablation.sh <gpu> <tw|nt> <base|e2e|residual> [base_ckpt]"; exit 1 ;;
esac

NAME="${SONG}-${METHOD}"

WANDB_DIR=./ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$DEVICE MUJOCO_EGL_DEVICE_ID=$DEVICE conda run -n cantabile --no-capture-output python train.py \
    --root-dir ./tmp \
    --warmstart-steps 5000 \
    --max-steps 5000000 \
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
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --mode "online" \
    --checkpoint-interval 500000 \
    --velocity-reward-coef $VEL \
    --onset-accuracy-reward-coef $ONSET \
    --name "$NAME" \
    $EXTRA_ARGS \
