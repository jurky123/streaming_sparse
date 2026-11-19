set -eu
source /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1/env.sh
d=/mnt/data/vbench_run_20261118/previews/light_speed_v1
sc="$1"; end="$2"; shift 2
while ! python - "$d/$end" <<'PY' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
assert len(d.get('times',[]))==4 or 'wall3' in d
PY
do sleep 3;done
for m in "$@"; do
 f=lfbench; if [ "$m" = native ];then f=densebench;fi
 REF_SUFFIX="_gpu$CUDA_VISIBLE_DEVICES" "$py" -u "$d/$f.py" "$m" "$sc" > "$d/job_${m}_${sc}_gpu$CUDA_VISIBLE_DEVICES.log" 2>&1
done
