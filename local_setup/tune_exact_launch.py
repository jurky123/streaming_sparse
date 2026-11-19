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
ref,rl=run(4,3);rows=[]
for w in [4,8,2]:
 for s in [1,2,3,4]:
  try:
   out,lse=run(w,s);equal=torch.equal(out,ref) and torch.equal(lse,rl)
   for _ in range(5):run(w,s)
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(60):run(w,s)
   torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/60
   row=dict(warps=w,stages=s,ms=ms,bitwise_equal=equal,max_abs=float((out-ref).abs().max()))
  except Exception as e:row=dict(warps=w,stages=s,error=str(e))
  rows.append(row);print(row,flush=True)
(r/'exact_launch_tuning.json').write_text(json.dumps(rows,indent=2))
