#!/usr/bin/env bash
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/migration/paired_router/status.py" "$@"
