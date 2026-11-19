"""Numerical/route gates for latest algorithm fast implementation."""
import os,json,torch
from pathlib import Path
from local_experiments.review_v1.ops import SummaryOps,RouteOps,Config
from local_experiments.review_v1.model import Model
from local_experiments.review_v1.runtime import SelectedAttention
from local_experiments.review_v1.fast_attention import attention
from local_experiments.review_v1.fast_route import GraphRoute
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4);torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt');m.load_state_dict(torch.load('artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt',weights_only=False)['model']);m.eval()
graph=GraphRoute();rows=[]
with torch.no_grad():
 for p in [8,216,432]:
  k=torch.randn(12,p,64,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);q=torch.randn(12,128,128,device='cuda',dtype=torch.bfloat16)
  s=SummaryOps.build(m[1],k,v);f,_=RouteOps.features(q,s);refscore=(f*m[1].theta[:,None,None]).sum(-1).reshape(12,8,16,p).mean(2)
  score=graph(q,s,m[1].theta);other_theta=m[2].theta+torch.tensor([.13,.27,.41,.59],device='cuda');score2=graph(q,s,other_theta)
  f2,_=RouteOps.features(q,s);ref2=(f2*other_theta[:,None,None]).sum(-1).reshape(12,8,16,p).mean(2)
  assert torch.equal(score,refscore) and torch.equal(score2,ref2),'Graph route must be bitwise equal and refresh layer weights'
  own=(torch.arange(2,device='cuda')+p-2)[None].expand(12,-1)
  selection=RouteOps.select(refscore[:,:2],own,Config(history_pages=360));selection['fixed_count']=max(1,int(p*.1))
  ref=SelectedAttention.run(q,k,v,s,m[1].basis,selection);out=attention(q,k,v,s,m[1].basis,selection)
  delta=out.float()-ref.float();nmse=float(delta.square().mean()/ref.float().square().mean().clamp_min(1e-12));err=float(delta.abs().max())
  row=dict(pages=p,route_bitwise_equal=True,max_abs=err,nmse=nmse,finite=bool(out.isfinite().all()))
  print(row,flush=True);assert row['finite'] and err<=.008 and nmse<=2e-5,row
  rows.append(row)
 result=dict(passed=True,rows=rows,scope='Same latest trained weights; FP32 route bitwise identical; mixed exact/full-mean/zero-sum-detail attention numeric tolerance')
 Path('/tmp/latest_fast_checks.json').write_text(json.dumps(result,indent=2))
