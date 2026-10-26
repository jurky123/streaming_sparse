#!/usr/bin/env bash
set -euo pipefail
cd /home/shuchenweng/jzk/streaming_sparse
if [[ "${1:-}" == "--once" ]]; then
 python migration/unified_router/opt10/status.py
else
 watch -n 10 python migration/unified_router/opt10/status.py
fi
