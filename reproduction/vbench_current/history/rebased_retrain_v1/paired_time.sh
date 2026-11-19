set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
idx="$1"; gpu="$2"
while [ ! -f "$exp/budget_$idx/reader.pt" ]; do
  if rg -q Traceback "$exp/train_$idx.log"; then exit 1; fi
  sleep 10
done
for round in 1 2 3; do
 if [ "$round" = 2 ]; then variants='trained original'; else variants='original trained'; fi
 for v in $variants; do
  CUDA_VISIBLE_DEVICES="$gpu" "$py" "$exp/video.py" "$idx" 0 "$v" "$round" >"$exp/timing_${idx}_${round}_${v}.log" 2>&1
 done
done
