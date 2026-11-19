set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
while [ "$(find "$exp/records" -name status.json | wc -l)" -ne 8 ]; do
  if rg -q 'Traceback' "$exp"/capture_*.log; then exit 1; fi
  sleep 5
done
job() {
 local gpu="$1" label="$2"
 shift 2
 CUDA_VISIBLE_DEVICES="$gpu" "$py" -u "$@" > "$exp/$label.log" 2>&1
}
train_and_video() {
 local gpu="$1" idx="$2"
 job "$gpu" "train_$idx" "$exp/train_ablation.py" "$idx" &&
 job "$gpu" "trained_${idx}_0" "$exp/video.py" "$idx" 0 trained &&
 job "$gpu" "trained_${idx}_1" "$exp/video.py" "$idx" 1 trained
}
train_and_video 0 11 &
train_and_video 1 21 &
job 2 dev_11 "$exp/train_ablation.py" 11 original &
job 3 dev_21 "$exp/train_ablation.py" 21 original &
for scene in 0 1; do
 (job "$((4+scene))" "dense_$scene" "$exp/video.py" 11 "$scene" dense) &
 (for idx in 11 21; do job "$((6+scene))" "original_${idx}_$scene" "$exp/video.py" "$idx" "$scene" original || exit 1; done) &
done
wait
for scene in 0 1; do job "$scene" "quality_$scene" "$exp/evaluate_video.py" "$scene" & done
wait
