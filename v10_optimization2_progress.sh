#!/bin/bash
cd /home/shuchenweng/jzk/streaming_sparse || exit 1
exec /usr/bin/python3 -m migration.unified_router.opt2_final_status "$@"
