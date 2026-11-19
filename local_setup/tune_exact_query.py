"""Tune existing exact kernel launch without changing its math."""
import os,time,json,torch
from pathlib import Path
os.chdir('/tmp/streaming_sparse-0/runtime');torch.set_num_threads(4)
from local_experiments.review_v1.exact_query_kernel import _attn_fwd
r=Path('/tmp/latest_kernel_optimization4');r.mkdir(exist_ok=True)
x=torch.load('/tmp/latest_kernel_optimization2/attention_inputs.pt',map_location='cuda',weights_only=True)
q=x['q'];k=x['k'];v=x['v'];h,n,d=q.shape;p=k.shape[1];count=x['selection']['fixed_count'];lut=x['selection']['ids'][...,:count].contiguous()[None]
qq=q.transpose(0,1).contiguous()[None];kk=k.flatten(1,2).transpose(0,1).contiguous()[None];vv=v.flatten(1,2).transpose(0,1).contiguous()[None]
def run(w,s,bq=64):
 out=torch.empty(qq.shape,device='cuda',dtype=torch.float32);lse=torch.empty(h,n,device='cuda')
 _attn_fwd[(n//bq,h)](qq,kk,vv,d**-.5,count,lut,lse,out,h,n,p*64,n//64,d,bq,64,HIGH_PRECISION_AV=True,num_warps=w,num_stages=s)
 return out,lse

ref,rl=run(4,2);rows=[]
for bq in [64,32,16]:
 for w,st in [(4,1),(4,2),(4,3),(8,2)]:
  try:
   actual,lse=run(w,st,bq);equal=torch.equal(actual,ref) and torch.equal(lse,rl)
   for _ in range(10):run(w,st,bq)
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(100):run(w,st,bq)
   torch.cuda.synchronize();ms=(time.perf_counter()-start)*10
   row=dict(bq=bq,warps=w,stages=st,ms=ms,bitwise_equal=equal,max_abs=float((actual-ref).abs().max()))
  except Exception as e:row=dict(bq=bq,warps=w,stages=st,error=str(e))
  rows.append(row);print(row,flush=True)
(r/'exact_query_tuning.json').write_text(json.dumps(rows,indent=2))
