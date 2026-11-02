#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$project_dir/streaming_sparse/.venv/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | head -n1 | cut -d, -f1 | tr -d ' ')"
  export CUDA_VISIBLE_DEVICES
  echo "Using physical GPU $CUDA_VISIBLE_DEVICES (most free memory); override with CUDA_VISIBLE_DEVICES=N." >&2
fi
exec "$project_dir/streaming_sparse/.venv/bin/python" "$@"
