#!/usr/bin/env bash
cd /home/shuchenweng/jzk/streaming_sparse
python - <<'PY'
import json
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/opt26_route_20261025')
v=[json.loads(p.read_text()) for p in root.glob('verify/*/layer*.json')]
print(f'路由块验证：{len(v)}/16；选页完全一致：{sum(r["exact"] for r in v)}')
print('720p / 45帧；Dense → Sparge32 → 路由优化；预热后5次取中位数')
for i in range(8):
 p=root/'e2e/720x1280'/f'case{i}';s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
 r=json.loads((p/'timing.json').read_text()).get('results',{}) if (p/'timing.json').exists() else {}
 msg='  '.join(f'{k}: {v["seconds"]:.3f}s' for k,v in r.items())
 print(f'case{i}: {s.get("status","未开始")} {s.get("variant","")} {s.get("repeat","")}  {msg}')
 if (p/'mismatch.json').exists():print('  输出不一致：',json.loads((p/'mismatch.json').read_text()))
PY
