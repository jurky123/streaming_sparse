#!/usr/bin/env bash
set -euo pipefail
cd /data/chenjiayu/zhikun/streaming_sparse
target=artifacts/cache_research_20261029
mkdir -p "$target/capture"
for scene in sailboat skateboard dog; do
  mkdir -p "$target/capture/$scene"
  if [[ ! -f "$target/capture/$scene/dense.generation.json" ]]; then
    bash local_setup/run.sh -m local_experiments.run_official_baseline --method dense \
      --inputs "artifacts/ar_official_baselines_20261029/$scene/shared_inputs.pt" \
      --output "$target/capture/$scene" \
      --research-config local_experiments/cache_research_capture.json \
      > "$target/capture/$scene/run.log" 2>&1
  fi
done
