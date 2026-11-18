"""Same-GPU native Dense ratios and bitwise acceptance of shared RoPE."""
import json
from pathlib import Path
import torch

def main():
 root=Path('artifacts/video_runtime_v1/fair_dense_v1');video=root/'video'
 results={p.parent.name:json.loads(p.read_text()) for p in video.glob('*/result.json')}
 def tensor(name):return torch.load(video/name/'latents.pt',map_location='cpu',weights_only=False)
 checks={}
 dense=tensor('gpu6_dense_original')
 for name in ['gpu6_dense_recent','gpu6_dense','gpu5_dense','gpu7_dense']:
  checks[name]=torch.equal(dense,tensor(name))
 checks['base_rope_ab']=torch.equal(tensor('gpu4_base_old'),tensor('gpu4_base_new'))
 rows=[]
 for gpu,tier in [(6,'base'),(7,'intersection'),(5,'step')]:
  name=f'gpu{gpu}_{tier}';old=Path('artifacts/video_runtime_v1/exact_pipeline_v1/video')/(tier+'_old')
  checks[tier]=torch.equal(tensor(name),torch.load(old/'latents.pt',map_location='cpu',weights_only=False))
  before=json.loads((old/'result.json').read_text())
  checks[tier+'_counters']=before['runtime_stats']['counters']==results[name]['runtime_stats']['counters']
  d=results[f'gpu{gpu}_dense']['median_seconds'];s=results[name]['median_seconds']
  rows.append(dict(tier=tier,gpu=gpu,dense_seconds=d,sparse_seconds=s,speedup=d/s))
 summary=dict(checks=checks,rows=rows,times={n:r['median_seconds'] for n,r in results.items()},rope=json.loads((root/'rope_module.json').read_text()))
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='rope'},indent=2));assert all(checks.values())
if __name__=='__main__':main()
