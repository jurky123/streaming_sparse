"""Consolidate the two requested optimization rounds and explicit acceptance decisions."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def main():
 root=ROOT/'artifacts/video_runtime_v1';r=root/'post_rope_round2';rows=[]
 for tier in ['base','intersection','step']:
  a=root/'post_rope_round1/video'/tier;b=r/'video'/tier
  before=json.loads((a/'result.json').read_text());after=json.loads((b/'result.json').read_text())
  x=torch.load(a/'latents.pt',map_location='cpu',weights_only=True).float();y=torch.load(b/'latents.pt',map_location='cpu',weights_only=True).float()
  chunks=[dict(chunk=i,bitwise=torch.equal(x[:,3*i:3*i+3],y[:,3*i:3*i+3])) for i in range(9)]
  row=dict(tier=tier,before_s=before['median_seconds'],after_s=after['median_seconds'],speedup=before['median_seconds']/after['median_seconds'],bitwise=torch.equal(x,y),nmse=((x-y).square().sum()/x.square().sum()).item(),max_abs=(x-y).abs().max().item(),counters_before=before['runtime_stats']['counters'],counters_after=after['runtime_stats']['counters'],chunks=chunks)
  quality=r/'video'/(tier+'_quality.json')
  if quality.exists():row['quality']=json.loads(quality.read_text())
  rows.append(row)
 (r/'acceptance.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k not in ('quality','chunks')} for x in rows],indent=2))
 files=['summary_moments.py','check_summary_moments.py','risk_online.py','check_risk_online.py','check_risk_online_route.py','deployment.py','accepted_tiers.py','profile_retained.py','quality_risk_online.py','report_post_rope.py']
 (r/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256((ROOT/'local_experiments/video_runtime_v1'/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n')
if __name__=='__main__':main()
