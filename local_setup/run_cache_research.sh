#!/usr/bin/env bash
set -euo pipefail
cd /data/chenjiayu/zhikun/streaming_sparse
target=artifacts/cache_research_20261029
bash local_experiments/run_cache_capture.sh
bash local_setup/run.sh -m local_experiments.screen_cache_research
bash local_setup/run.sh -m local_experiments.check_cache_prototype
bash local_setup/run.sh -m local_experiments.run_cache_video_suite
if [[ ! -f "$target/diagnostic/dense.generation.json" ]]; then
  bash local_setup/run.sh -m local_experiments.run_official_baseline --method dense \
    --inputs artifacts/ar_official_baselines_20261029/sailboat/shared_inputs.pt \
    --output "$target/diagnostic" --research-config local_experiments/cache_research_diagnose.json
fi
bash local_setup/run.sh -m local_experiments.run_cache_followup
if [[ ! -f "$target/capture_fullhead/dense.generation.json" ]]; then
  bash local_setup/run.sh -m local_experiments.run_official_baseline --method dense \
    --inputs artifacts/ar_official_baselines_20261029/sailboat/shared_inputs.pt \
    --output "$target/capture_fullhead" --research-config local_experiments/cache_research_capture_fullhead.json
fi
bash local_setup/run.sh -m local_experiments.benchmark_cache_interleaved
bash local_setup/run.sh -m local_experiments.probe_rope_statistics
bash local_setup/run.sh -m local_experiments.analyze_cache_research
bash local_setup/run.sh -m local_experiments.analyze_cache_followup
bash local_setup/run.sh -m local_experiments.audit_cache_research
bash local_setup/run.sh -m local_experiments.build_cache_report
