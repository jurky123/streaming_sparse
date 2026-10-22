#!/usr/bin/env bash
# Reconstruct the original data tasks on ONE GPU. No training starts here.
set -euo pipefail
cd "$(dirname "$0")/.."
source streaming_sparse/.venv/bin/activate
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
case "${1:-replay}" in
  replay)
    python -m streaming_sparse.v08.collect --split train
    python -m streaming_sparse.v08.collect --split val
    ;;
  legacy-qkv)
    # The unchanged protocol hashes this absolute root; do not silently edit it.
    python - <<'PY'
from pathlib import Path
from streaming_sparse.v08.common import protocol, ROOT
assert Path(protocol()['warmstart']['legacy_qkv_root']).resolve() == (ROOT/'streaming_sparse/v04/data/full').resolve(), 'Keep the original compatible absolute path; see REBUILD.md'
PY
    target=streaming_sparse/v04/data/full
    if [[ -e "$target/events.jsonl" ]]; then
      echo 'Refusing to append a duplicate legacy capture; inspect the existing directory.' >&2
      exit 1
    fi
    python -m streaming_sparse.v02.capture --mode capture \
      --prompts streaming_sparse/v04/configs/capture_prompts.txt --limit 24 --frames 12 \
      --seeds 0 1 --layers 4 14 24 --heads 0 1 2 3 4 5 6 7 8 9 10 11 \
      --query-tile 16 --tiles-per-event 8 --sampling-seed 2026 \
      --backend sdpa --output "$target"
    python - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('streaming_sparse/v03/configs/protocol.json').read_text())
Path('streaming_sparse/v04/data/full/splits.json').write_text(json.dumps(p['splits'],indent=2)+'\n')
from streaming_sparse.v04.data import SnapshotData
for split in ('train','val','test'):
    d=SnapshotData('streaming_sparse/v04/data/full',split)
    print(split, len(d.groups), 'complete snapshots')
PY
    ;;
  *) echo 'Usage: bash migration/recollect.sh [replay|legacy-qkv]' >&2; exit 2 ;;
esac
