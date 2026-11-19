set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
d="$prev/../conditional_budget_v1"; t="$1"; i="$2"
for rep in 1 2 3;do
 order='fixed table'; if [ "$rep" = 2 ];then order='table fixed';fi
 for method in $order;do
  if [ "$method" = fixed ];then folder="$exp";idx="$i";else folder="$d";idx="$t";fi
  "$py" "$folder/video.py" "$idx" 0 original "cond$rep" > "$d/timing_${t}_${method}_$rep.log" 2>&1
  cp "$folder/0_${idx}_original_repeatcond$rep.json" "$d/time_${t}_${method}_$rep.json"
 done
done
