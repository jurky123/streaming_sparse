set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
for scene in 0 1; do
 for idx in 11 21; do
  for variant in reader_only trained; do
   while ! ffprobe -v error -show_entries format=duration "$exp/${scene}_${idx}_${variant}.mp4" >/dev/null 2>&1; do sleep 5; done
  done
  "$py" -c 'import sys;from visual_review import sheets;sheets(int(sys.argv[1]),int(sys.argv[2]),False)' "$scene" "$idx"
 done
done
