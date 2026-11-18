#!/usr/bin/env bash
set -euo pipefail
folder=$(cd "$(dirname "$0")" && pwd)
cd "$folder/../../.."
export PYTHONPATH=.
py=/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
compile() {
  /usr/local/cuda/bin/nvcc -arch=sm_80 -O3 --fmad=false -lineinfo -Xcompiler=-fPIC -shared "$folder/$1.cu" -o "/tmp/jzk_fixed15_$1.so"
}
compile paired_k
case "$1" in
  module)
    for pages in 216 288 360 432; do "$py" "$folder/module.py" "$pages"; done
    ;;
  video) "$py" "$folder/paired_video.py" --pairs 6 ;;
  *) echo "Usage: $0 module|video" >&2; exit 2 ;;
esac
