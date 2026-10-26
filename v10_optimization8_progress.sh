#!/usr/bin/env bash
cd /home/shuchenweng/jzk/streaming_sparse || exit 1
if [ "${1:-}" = '--once' ]; then
 python migration/unified_router/opt8/status.py
else
 watch -n 5 python migration/unified_router/opt8/status.py
fi
