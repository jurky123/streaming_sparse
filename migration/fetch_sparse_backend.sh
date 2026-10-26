#!/usr/bin/env bash
# Retrieve source headers only; no models, datasets or experiment outputs.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
backend_dir="$repo_root/streaming_sparse/v10/experiments/opt24_attention_backends_20261025/vendor/sparge"
backend_revision=ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a
if [[ "${1:-}" == --check-only ]]; then
  test "$(git -C "$backend_dir" rev-parse HEAD)" = "$backend_revision"
  test -z "$(git -C "$backend_dir" status --porcelain)"
  test -f "$backend_dir/csrc/qattn/attn_utils.cuh" || test -d "$backend_dir/csrc/qattn"
  echo "SpargeAttn source revision verified: $backend_revision"
  exit 0
fi
if [[ -e "$backend_dir" ]]; then
  bash "$0" --check-only
  exit 0
fi
mkdir -p -- "$(dirname -- "$backend_dir")"
git clone https://github.com/thu-ml/SpargeAttn.git "$backend_dir"
git -C "$backend_dir" checkout --detach "$backend_revision"
bash "$0" --check-only
