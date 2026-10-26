#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -m migration.unified_router.opt18.status
exec python -m migration.unified_router.opt18.frontier_status
