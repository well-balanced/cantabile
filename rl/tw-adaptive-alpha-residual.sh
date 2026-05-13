#!/bin/bash

DEVICE=${1:-0}
WANDB_DIR=./ MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$DEVICE XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$DEVICE \
conda run -n cantabile --no-capture-output python train.py \
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
    --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --mode "online" \
    --checkpoint-interval 500000 \
    --velocity-reward-coef 0.2 \
    --onset-accuracy-reward-coef 0.5 \
    --residual-alpha 0.1 \
    --residual-action-mode fingers_only \
    --base-checkpoint ./tmp/checkpoints/tw-e2e-v02-o05/checkpoint_5000000.flax \
    --adaptive-alpha \
    --alpha-max 1.0 \
    --name "tw-adaptive-alpha-residual"
