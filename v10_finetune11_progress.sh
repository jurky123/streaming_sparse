#!/usr/bin/env bash
set -euo pipefail
cd /home/shuchenweng/jzk/streaming_sparse
if [[ "${1:-}" == "--once" ]]; then
  python -m migration.unified_router.opt11.status
else
  watch -n 10 'python -m migration.unified_router.opt11.status'
fi
