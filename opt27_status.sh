#!/usr/bin/env bash
cd /home/shuchenweng/jzk/streaming_sparse
python - <<'PY'
import json
from pathlib import Path
root=Path('streaming_sparse/v10/experiments/opt27_fused_refine_20261025')
print('Opt27：精筛融合未胜出；正式验证 Q/KV 数据准备融合')
for phase in ['scan','mass','shared']:
 d=[json.loads(p.read_text()) for p in (root/phase).glob('*/*.json')]
 print(phase, '完成组数',sum(x['completed'] for x in d),'配置数',sum(len(x['rows']) for x in d))
print('正式块复测：',len(list((root/'confirm').glob('*/*.json'))),'/16')
for i in range(8):
 p=root/'e2e/720x1280'/f'case{i}';s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {};r=json.loads((p/'timing.json').read_text()).get('results',{}) if (p/'timing.json').exists() else {}
 print(f'case{i}: {s.get("status","未开始")} {s.get("variant","")} {s.get("repeat","")}  '+ '  '.join(f'{k}: {v["seconds"]:.3f}s' for k,v in r.items()))
 if (p/'mismatch.json').exists():print('输出不一致：',json.loads((p/'mismatch.json').read_text()))
PY
