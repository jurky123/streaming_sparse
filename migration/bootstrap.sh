#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.cache/pip}"
python3.10 -m venv streaming_sparse/.venv
source streaming_sparse/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r migration/requirements.lock.txt
python -m pip install --no-deps -e .
python migration/verify.py
