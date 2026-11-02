#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
output_dir="${1:-artifacts/ar_official_baselines_20261029}"
bash local_setup/run.sh -m local_experiments.run_official_suite --output "$output_dir"
bash local_setup/run.sh -m local_experiments.analyze_official_baselines "$output_dir" --device cuda
bash local_setup/run.sh -m local_experiments.audit_official_baselines "$output_dir"
bash local_setup/run.sh -m local_experiments.build_official_report "$output_dir"
