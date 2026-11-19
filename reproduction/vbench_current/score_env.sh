#!/usr/bin/env bash
set -euo pipefail
: "${VBENCH_RUN:?export an absolute run directory}"
code=$(cd "$(dirname "$0")" && pwd)
python3.10 -m venv "$VBENCH_RUN/eval_env_local"
py="$VBENCH_RUN/eval_env_local/bin/python"
"$py" -m pip install 'pip<25' wheel setuptools
"$py" -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
# The upstream VBench wheel has a different source snapshot. Import our pinned checkout.
rg -vi '^(detectron2|vbench|pip|setuptools|wheel)==' "$code/scoring.versions" > "$VBENCH_RUN/scoring-pypi.txt"
"$py" -m pip install --no-deps -r "$VBENCH_RUN/scoring-pypi.txt"
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 "$py" -m pip install --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/detectron2.git@a9c0821a12ad353fb2a96f019515990d5460c5ac'
"$py" -m pip install --target "$VBENCH_RUN/eval_deps" --no-deps accelerate==0.26.1 av==12.3.0 decorator==4.4.2 dreamsim==0.2.1 imageio-ffmpeg==0.6.0 moviepy==1.0.3 open-clip-torch==2.24.0 peft==0.7.1 proglog==0.1.10 scenedetect==0.6.5.2
