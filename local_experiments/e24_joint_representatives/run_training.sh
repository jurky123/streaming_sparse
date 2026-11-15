#!/usr/bin/env bash
set -euo pipefail
cd /home/shuchenweng/jzk/streaming_sparse
PY=/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while [ "$(find artifacts/e24_joint_representatives_v1/trajectories -name complete.json | wc -l)" -lt 14 ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=6 "$PY" -m local_experiments.e24_joint_representatives.train --fit > /tmp/e24_fit.log 2>&1
(CUDA_VISIBLE_DEVICES=0 "$PY" -m local_experiments.e24_joint_representatives.train --group A --seed 17 > /tmp/e24_A17.log 2>&1 && CUDA_VISIBLE_DEVICES=0 "$PY" -m local_experiments.e24_joint_representatives.train --group C --seed 17 > /tmp/e24_C17.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 "$PY" -m local_experiments.e24_joint_representatives.train --group A --seed 29 > /tmp/e24_A29.log 2>&1 && CUDA_VISIBLE_DEVICES=1 "$PY" -m local_experiments.e24_joint_representatives.train --group C --seed 29 > /tmp/e24_C29.log 2>&1) &
CUDA_VISIBLE_DEVICES=2 "$PY" -m local_experiments.e24_joint_representatives.train --group B --seed 17 > /tmp/e24_B17.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 "$PY" -m local_experiments.e24_joint_representatives.train --group B --seed 29 > /tmp/e24_B29.log 2>&1 &
wait
