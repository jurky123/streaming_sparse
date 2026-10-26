#!/usr/bin/env bash
cd /home/shuchenweng/jzk/streaming_sparse
python - <<'PY'
import json
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/opt25_720p_e2e_20261025/720x1280')
print('720p / 45帧 / 20%预算；计时排除首次编译和CUDA Graph捕获')
for i in range(8):
 p=root/f'case{i}';s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
 r=json.loads((p/'timing.json').read_text()).get('results',{}) if (p/'timing.json').exists() else {}
 times='  '.join(f'{k}: {v["seconds"]:.3f}s' for k,v in r.items())
 print(f'case{i}: {s.get("status","未开始")} {s.get("variant","")} {s.get("repeat","")}  {times}')
PY
