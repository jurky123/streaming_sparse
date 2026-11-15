#!/usr/bin/env bash
set -euo pipefail
cd /home/shuchenweng/jzk/streaming_sparse
while [ "$(find artifacts/e24_joint_representatives_v1/training -name complete.json | wc -l)" -lt 6 ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.e24_joint_representatives.finalize > /tmp/e24_finalize.log 2>&1
