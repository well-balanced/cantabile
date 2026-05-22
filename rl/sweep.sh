#!/bin/bash
# vel_aware hyperparameter sweep — vel_coef × onset_coef (twinkle, seed 0)
#
# Grid:
#   vel_coef:   0.0  0.1  0.2  0.5
#   onset_coef: 0.0  0.2  0.5  1.0
#
# Run name suffix encodes values: v{vel*10}-o{onset*10}
#   e.g. vel=0.2, onset=0.5 → v02-o05
#
# Launch Round 1 (GPU 0-7), wait for completion, then launch Round 2 (GPU 0-7).

# ── Round 1 ───────────────────────────────────────────────────────────────────

bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 0 --vel 0.0 --onset 0.0 --suffix v00-o00 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 1 --vel 0.0 --onset 0.2 --suffix v00-o02 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 2 --vel 0.0 --onset 0.5 --suffix v00-o05 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 3 --vel 0.0 --onset 1.0 --suffix v00-o10 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 4 --vel 0.1 --onset 0.0 --suffix v01-o00 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 5 --vel 0.1 --onset 0.2 --suffix v01-o02 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 6 --vel 0.1 --onset 0.5 --suffix v01-o05 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 7 --vel 0.1 --onset 1.0 --suffix v01-o10 &

wait

# ── Round 2 ───────────────────────────────────────────────────────────────────

bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 0 --vel 0.2 --onset 0.0 --suffix v02-o00 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 1 --vel 0.2 --onset 0.2 --suffix v02-o02 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 2 --vel 0.2 --onset 0.5 --suffix v02-o05 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 3 --vel 0.2 --onset 1.0 --suffix v02-o10 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 4 --vel 0.5 --onset 0.0 --suffix v05-o00 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 5 --vel 0.5 --onset 0.2 --suffix v05-o02 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 6 --vel 0.5 --onset 0.5 --suffix v05-o05 &
bash run.sh --method vel_aware --song twinkle --seed 0 --gpu 7 --vel 0.5 --onset 1.0 --suffix v05-o10 &

wait
