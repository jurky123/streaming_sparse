#!/usr/bin/env bash
# Read-only unified terminal dashboard; works from any directory.
set -eu
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -B "$script_dir/training_status.py" "$@"
