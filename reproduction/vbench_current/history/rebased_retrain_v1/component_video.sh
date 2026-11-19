set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
idx="$1"; gpu="$2"
for item in 'C_only summary.pt' 'reader_only reader.pt'; do
 read -r variant artifact <<< "$item"
 while [ ! -f "$exp/budget_$idx/$artifact" ]; do
   if rg -q Traceback "$exp/train_$idx.log"; then exit 1; fi
   sleep 10
 done
 for scene in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" "$py" "$exp/video.py" "$idx" "$scene" "$variant" >"$exp/component_${idx}_${variant}_${scene}.log" 2>&1
  ABLATION_COMPONENT="$variant" CUDA_VISIBLE_DEVICES="$gpu" "$py" "$exp/evaluate_video.py" "$scene" "$idx" >"$exp/component_metric_${idx}_${variant}_${scene}.log" 2>&1
 done
done
