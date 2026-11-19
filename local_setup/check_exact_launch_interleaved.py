"""Tune existing exact kernel launch without changing its math."""
import os,time,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4)
from local_experiments.gap_exact_round_kernel import _attn_fwd
r=Path('/tmp/latest_kernel_optimization3');r.mkdir(exist_ok=True)
x=torch.load('/tmp/latest_kernel_optimization2/attention_inputs.pt',map_location='cuda',weights_only=True)
q=x['q'];k=x['k'];v=x['v'];h,n,d=q.shape;p=k.shape[1];count=x['selection']['fixed_count'];lut=x['selection']['ids'][...,:count].contiguous()[None]
qq=q.transpose(0,1).contiguous()[None];kk=k.flatten(1,2).transpose(0,1).contiguous()[None];vv=v.flatten(1,2).transpose(0,1).contiguous()[None]
def run(w,s):
 out=torch.empty(qq.shape,device='cuda',dtype=torch.float32);lse=torch.empty(h,n,device='cuda')
 _attn_fwd[(n//64,h)](qq,kk,vv,d**-.5,count,lut,lse,out,h,n,p*64,n//64,d,64,64,HIGH_PRECISION_AV=True,num_warps=w,num_stages=s)
 return out,lse

rows=[]
for _ in range(100):run(4,3)
for repeat in range(4):
 for stage in ([3,2,4] if repeat%2==0 else [4,2,3]):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record()
  for _ in range(100):run(4,stage)
  b.record();b.synchronize()
  row=dict(repeat=repeat,stages=stage,device_ms=a.elapsed_time(b)/100);rows.append(row);print(row,flush=True)
(r/'exact_launch_interleaved.json').write_text(json.dumps(rows,indent=2))
