#!/bin/bash

DEVICE=${1:-0}
STEP=${2:-1}

case $STEP in
    1|2|3) ENV="RoboPianist-debug-Firststep${STEP}-v0" ;;
    *) echo "Usage: bash custom.sh <gpu> <1|2|3>"; exit 1 ;;
esac

NAME="fs${STEP}-vel-onset"

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
    --disable-fingering-reward \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --mode "online" \
    --velocity-reward-coef 0.5 \
    --onset-accuracy-reward-coef 0.5 \
    --name "$NAME" \
