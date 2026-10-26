#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec streaming_sparse/.venv/bin/python -m migration.unified_router.opt20.status "$@"
