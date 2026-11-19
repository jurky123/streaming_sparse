set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
d="$prev/../conditional_budget_v1"
for pass in 3 4; do
 variant=trained; if [ "$pass" = 4 ];then variant=refit;fi
 for sc in 0 1;do for idx in 10 15; do while ! ffprobe -v error -show_entries format=duration "$d/${sc}_${idx}_$variant.mp4" >/dev/null 2>&1; do sleep 5;done;done;done
 "$py" "$d/report.py"
 "$py" "$d/boards.py" "$pass"
done
