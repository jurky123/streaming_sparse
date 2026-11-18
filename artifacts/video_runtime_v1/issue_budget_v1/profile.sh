#!/usr/bin/env bash
set -eu
dir=$(dirname "$0")
metrics=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_pipe_tensor.sum,smsp__inst_executed.sum,smsp__warps_eligible.avg.per_cycle_active,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active
for p in "$@"; do
  ncu --profile-from-start off --clock-control "${CLOCK_CONTROL:-base}" --metrics "$metrics" --csv --log-file "$dir/p${p}_${CLOCK_CONTROL:-base}.csv" /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python "$dir/measure.py" "$p"
done
