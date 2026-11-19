#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3.10 -m venv .venv-vbench-gen
py=.venv-vbench-gen/bin/python
"$py" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
rg -vi '^(self_forcing|torch|torchvision|pip|setuptools|wheel)==' reproduction/vbench_current/generation.versions > .venv-vbench-gen/packages.txt
"$py" -m pip install --no-deps -r .venv-vbench-gen/packages.txt
"$py" -m pip install --no-deps -e .
