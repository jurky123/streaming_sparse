#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 migration/unified_router/profile_status.py "$@"
