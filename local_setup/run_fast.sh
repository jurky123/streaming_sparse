#!/usr/bin/env bash
# Stage a disposable local runtime; keep experiment outputs in the repository.
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cache_dir="${STREAMING_SPARSE_CACHE:-/tmp/streaming_sparse-${UID}/runtime}"
if [[ "$cache_dir" == "$source_dir" || "$cache_dir" == / ]]; then
  echo 'Cache must be a separate directory.' >&2
  exit 1
fi
mkdir -p "$cache_dir"
revision="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "${1:-}" == --refresh ]]; then
  shift
  refresh=1
else
  refresh=0
fi
if [[ "$refresh" == 1 || ! -f "$cache_dir/.cache_revision" || "$(cat "$cache_dir/.cache_revision")" != "$revision" ]]; then
# Copy tracked source from the worktree; omit historical run artifacts.
git -C "$source_dir" ls-files -z | rsync -a --from0 --files-from=- --exclude='streaming_sparse/v08/runs/***' "$source_dir/" "$cache_dir/"
mkdir -p "$cache_dir/streaming_sparse" "$cache_dir/wan_models/Wan2.1-T2V-1.3B" "$cache_dir/checkpoints"
rsync -a "$source_dir/streaming_sparse/.venv/" "$cache_dir/streaming_sparse/.venv/"
rsync -a "$source_dir/checkpoints/self_forcing_dmd.pt" "$cache_dir/checkpoints/"
for asset in config.json diffusion_pytorch_model.safetensors Wan2.1_VAE.pth; do
  rsync -a "$source_dir/wan_models/Wan2.1-T2V-1.3B/$asset" "$cache_dir/wan_models/Wan2.1-T2V-1.3B/"
done
printf '%s\n' "$revision" > "$cache_dir/.cache_revision"
fi
# Weights must stay inside the runtime root for the publisher's path checks.
if [[ -L "$cache_dir/artifacts" ]]; then
  if [[ "$(readlink "$cache_dir/artifacts")" != "$source_dir/artifacts" ]]; then
    echo 'Unexpected artifacts symlink in cache.' >&2
    exit 1
  fi
  unlink "$cache_dir/artifacts"
fi
if [[ ! -f "$cache_dir/.weights_ready" ]]; then
  "$cache_dir/streaming_sparse/.venv/bin/python" "$cache_dir/reproduction/restore_weights.py" --destination "$cache_dir"
  touch "$cache_dir/.weights_ready"
fi
sync_outputs() {
  rsync -a "$cache_dir/artifacts/" "$source_dir/artifacts/"
}
trap sync_outputs EXIT
export PYTHONPATH="$cache_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd "$cache_dir"
echo "Local runtime: $cache_dir; outputs sync on exit: $source_dir/artifacts" >&2
"$cache_dir/streaming_sparse/.venv/bin/python" "$@"
