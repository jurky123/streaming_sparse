set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
d="$prev/../light_speed_v1"; export PYTHONPATH="$prev/../conditional_budget_v1:$PYTHONPATH"
scene="$1";shift
for m in "$@";do
 case "$m" in
 legacy|light) args=("$d/lfbench.py" "$m" "$scene");;
 10|15) args=("$d/tablebench.py" "$m" "$scene" original run);;
 *) args=("$d/densebench.py" "$m" "$scene");;
 esac
 "$py" -u "${args[@]}" >"$d/job_${m}_${scene}.log" 2>&1 || exit 1
done
