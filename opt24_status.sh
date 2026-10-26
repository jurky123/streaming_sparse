#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec streaming_sparse/.venv/bin/python -m migration.unified_router.opt24.status
