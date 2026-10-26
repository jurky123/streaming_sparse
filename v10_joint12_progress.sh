#!/usr/bin/env bash
set -euo pipefail
cd /home/shuchenweng/jzk/streaming_sparse
if [[ "${1:-}" == "--once" ]]; then
  python -m migration.unified_router.opt12.status
elif [[ "${1:-}" == "--plot" ]]; then
  streaming_sparse/.venv/bin/python -m migration.unified_router.opt12.plot
else
  watch -n 10 'python -m migration.unified_router.opt12.status'
fi
