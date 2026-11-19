set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
d="$prev/../conditional_budget_v1"; export PYTHONPATH="$d:$PYTHONPATH"
idx="$1";gpu="$2"
"$py" "$d/train_conditional.py" "$idx" >"$d/train_$idx.log" 2>&1
for s in 0 1;do "$py" "$d/video.py" "$idx" "$s" refit >"$d/refit_${idx}_$s.log" 2>&1;done
