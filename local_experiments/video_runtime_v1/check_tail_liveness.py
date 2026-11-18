"""Same-input tail outputs, compilation resources and CUDA Graph latency."""
import argparse,json,statistics
import torch,triton as tr
from .weights import ROOT
from ..review_v1.tuned_attention import _tail
from ..high_sparse_runtime import _budget_mask
from .tail_liveness import _tail_late
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--split',action='store_true');parser.add_argument('--query32',action='store_true');parser.add_argument('--convert',action='store_true');opts=parser.parse_args()
 torch.set_num_threads(4)
 root=ROOT/'artifacts/video_runtime_v1/tail_liveness_v1';s=torch.load(root/'input.pt',weights_only=True);q=s['q'].cuda();h=12;n=q.shape[1];d=128;rows=[]
 for p in (216,288,360,432):
  keys,mean,detail=[s['summary'][k][:,:p].cuda().contiguous() for k in ('keys','mean','detail')]
  ids=(s['ids'][...,:p//10]%p).cuda().contiguous()[None];bits=torch.empty(h,n//64,16,device='cuda',dtype=torch.uint32)
  _budget_mask[(n//64,h)](ids,bits,n//64,p//10,tr.next_power_of_2(p//10),num_warps=4)
  from .tail_split import _tail_part
  from .tail_convert import _tail_convert
  outputs={name:(torch.empty(h,n,d,device='cuda'),torch.empty(h,n,8,device='cuda'),torch.empty(h,n,device='cuda')) for name in ('old','late')};graphs={};resources={}
  for name,kernel in [('old',_tail),('late',_tail_late)]:
   bq=32 if opts.query32 and name=='late' else 64
   if opts.query32 and name=='late':kernel=_tail
   if opts.convert and name=='late':kernel=_tail_convert
   args=(q,keys,mean,detail,bits,*outputs[name],p,n,h,d)
   def run():
    if opts.split and name=='late':
     first=_tail_part[(n//64,h)](*args,BQ=64,BR=64,BF16_AV=True,PART=0,num_warps=4,num_stages=2,enable_fp_fusion=False)
     second=_tail_part[(n//64,h)](*args,BQ=64,BR=64,BF16_AV=True,PART=1,num_warps=4,num_stages=2,enable_fp_fusion=False)
     return first,second
    return kernel[(n//bq,h)](*args,BQ=bq,BR=64,BF16_AV=True,num_warps=4,num_stages=2,enable_fp_fusion=False)
   compiled=run();torch.cuda.synchronize()
   if isinstance(compiled,tuple):resources[name]=[dict(registers=c.n_regs,spills=c.n_spills,shared=c.metadata.shared) for c in compiled]
   else:resources[name]=dict(registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared)
   if p==432 and not opts.split and not opts.query32 and not opts.convert:(root/(name+'.ptx')).write_text(compiled.asm['ptx'])
   for _ in range(3):run()
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):run()
   graphs[name]=g
  checks=[dict(equal=torch.equal(a,b),max_abs=float((a-b).abs().max())) for a,b in zip(outputs['old'],outputs['late'])];times={name:[] for name in graphs}
  for rep in range(20):
   for name in (('old','late') if rep%2==0 else ('late','old')):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(10):graphs[name].replay()
    b.record();b.synchronize();times[name].append(a.elapsed_time(b)/10)
  med={name:statistics.median(v) for name,v in times.items()};row=dict(pages=p,contracts=checks,resources=resources,median_ms=med,speedup=med['old']/med['late'],times=times);rows.append(row);print({k:v for k,v in row.items() if k!='times'},flush=True)
 (root/('convert_result.json' if opts.convert else 'query32_result.json' if opts.query32 else 'split_result.json' if opts.split else 'result.json')).write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
