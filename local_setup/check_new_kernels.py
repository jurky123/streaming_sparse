import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,'/tmp/streaming_sparse_baselines_deps')
root=Path('/home/shuchenweng/jzk/streaming_sparse/external_baselines')
sys.path.insert(0,str(root/'PISA'));sys.path.insert(0,str(root/'Sana/techniques/sparse_backends'))
import torch
rows=[]
for name in ['pisa','solattn']:
 try:
  if name=='pisa':
   from piecewise_attn import piecewise_sparse_attention as fn
   q=torch.zeros(1,2,256,128,device='cuda',dtype=torch.bfloat16);q[:,:,-64:]=torch.randn_like(q[:,:,-64:]);k=torch.randn_like(q);v=torch.randn_like(q)
   y=fn(q,k,v,density=1.)
  else:
   from sol_attn import sol_attn as fn
   q=torch.zeros(1,256,2,128,device='cuda',dtype=torch.bfloat16);q[:,-64:]=torch.randn_like(q[:,-64:]);k=torch.randn_like(q);v=torch.randn_like(q)
   y=fn(q,k,v,tau=1.,sink_tokens=256)
   q,k,v,y=[x.transpose(1,2) for x in (q,k,v,y)]
  ref=torch.nn.functional.scaled_dot_product_attention(q,k,v)
  error=(y[:,:,-64:]-ref[:,:,-64:]).abs().max().item();assert torch.isfinite(y).all();assert error<0.05,error
  rows.append(dict(method=name,status='passed',dense_max_abs_error=error))
 except Exception as e:
  traceback.print_exc();rows.append(dict(method=name,status='failed',error=repr(e)))
 print(json.dumps(rows[-1]),flush=True)
Path('/tmp/new_kernel_checks.json').write_text(json.dumps(rows,indent=2))
