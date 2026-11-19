set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
idx="$1"
gpu="$2"
for spec in 'C_only summary.pt' 'reader_only reader.pt'; do
 read -r variant artifact <<< "$spec"
 while [ ! -f "$exp/budget_$idx/$artifact" ]; do
   if rg -q Traceback "$exp/train_$idx.log"; then exit 1; fi
   sleep 10
 done
 CUDA_VISIBLE_DEVICES="$gpu" "$py" "$exp/train_ablation.py" "$idx" "$variant" >"$exp/${variant}_$idx.log" 2>&1
done
