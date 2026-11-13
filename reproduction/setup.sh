#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
# Isolated venv; do not execute the legacy migration inventory audit.
if [[ ! -x streaming_sparse/.venv/bin/python ]]; then
  python3.10 -m venv streaming_sparse/.venv
fi
py=streaming_sparse/.venv/bin/python
"$py" -m pip install --upgrade pip setuptools wheel
"$py" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$py" -m pip install -r migration/requirements.lock.txt
"$py" -m pip install --no-deps -e .
"$py" -m unittest local_experiments.review_v1.test_ops
python3 reproduction/restore_weights.py
