"""Measure identical selected sets visited in ranked and physical page order."""
import ctypes as C
import json
import os
import sys
from pathlib import Path
from statistics import median
import torch
from local_experiments.video_runtime_v1.fixed15_attention import library
from local_experiments.video_runtime_v1.fixed15_layout import encode

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'operand_delivery_v1'))
from capture import q,pk,pv,ids,summary,basis,keys,mean,detail,p,hp,count,times,timed
tiles=(p+15)//16
encode[(12,tiles,42)](keys,mean,detail,summary,p,hp,tiles,DIRECT_MASS=True,num_warps=4)
physical=ids.clone()
physical[...,25:]=ids[...,25:].sort(dim=-1).values
assert torch.equal(ids.sort().values,physical.sort().values)
assert torch.equal(ids[...,1:25],torch.arange(24,device=q.device).expand_as(ids[...,1:25]))

def setup(order,lib,packed_k,packed_v):
    out=torch.empty_like(q)
    flags=torch.empty((12,72),device=q.device,dtype=torch.int32)
    ptr=[t.data_ptr() for t in (q,packed_k,packed_v,order,out,summary,basis,flags)]
    def invoke():
        assert lib.run(*ptr,p,count,torch.cuda.current_stream().cuda_stream)==0
    invoke()
    return invoke,out,flags

def locality(order):
    suffix=order[...,25:].cpu()
    unique=[len(torch.unique(step)) for head in suffix for step in head.T]
    return dict(mean_dist=float(suffix.diff().abs().float().mean()),mean_distinct=sum(unique)/len(unique))

head_major=bool(os.getenv('HEAD_MAJOR'))
common=bool(os.getenv('COMMON_SUFFIX'))
if common:
    # A diagnostic: all queries select the same contiguous non-anchor history pages.
    physical[...,25:]=torch.arange(24,24+count-25,device=q.device)
    assert torch.all(physical.sort().values.diff()!=0)

a,old,of=setup(ids,library(),pk,pv)
if head_major:
    def transpose(src):
        dst=src.new_empty((12,432,8192))
        dst[:,:p].copy_(src.view(p,12,8192).transpose(0,1))
        return dst
    hk,hv=transpose(pk),transpose(pv)
    lib=C.CDLL('/tmp/jzk_visit_head.so')
    lib.run.argtypes=library().run.argtypes
    b,new,nf=setup(ids,lib,hk,hv)
    assert torch.equal(old.view(torch.uint8),new.view(torch.uint8))
else:
    b,new,nf=setup(physical,library(),pk,pv)
assert torch.equal(of,nf) and not of.any()
if os.getenv('COUNTERS'):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    a();b()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    sys.exit()
x,y=times([timed(a),timed(b)])
diff=new.float()-old.float()
row=dict(p=p,exact=count,base_ms=median(x),new_ms=median(y),speedup=median(i/j for i,j in zip(x,y)),
         nmse=float(diff.square().sum()/old.float().square().sum()),max_abs=float(diff.abs().max()),
         changed=int((old!=new).sum()),base_locality=locality(ids),candidate_locality=locality(ids if head_major else physical),samples=[x,y])
(root/f"{'common_' if common else 'head_' if head_major else ''}p{p}{os.getenv('RUN_SUFFIX','')}.json").write_text(json.dumps(row,indent=2))
print({k:v for k,v in row.items() if k!='samples'})
