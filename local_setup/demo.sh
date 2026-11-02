#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/local_setup/run.sh" -m streaming_sparse.capture.capture_runner \
  --mode dense --prompts local_setup/prompt.txt --limit 1 \
  --frames 3 --backend sdpa --save-video --output local_setup/results/demo "$@"
