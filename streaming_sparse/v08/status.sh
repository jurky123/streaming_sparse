#!/usr/bin/env bash
set -euo pipefail
cavir_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$cavir_root"
exec "$cavir_root/streaming_sparse/.venv/bin/python" -m streaming_sparse.v08.monitor.status "$@"
