from pathlib import Path
import sys
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import *
from kernel import specialized
from local_experiments.high_sparse_runtime import _budget_mask
import triton
assert torch.equal(ids[...,1:25],torch.arange(24,device='cuda').expand_as(ids[...,1:25]))
variants=[('control',False,False),('skip',False,True),('prefix',True,False),('both',True,True)]
def run(prefix,skip,kernel=specialized):
 return kernel[(72,12)](q,keys,mean,detail,None,None,None,None,p,n,h,D,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=new,HEAD_PAGES=hp,K=k,V=v,IDS=ids,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=count,QK_SCALE=128**-.5,PREFIX=prefix,SKIP=skip,num_warps=4,num_stages=2,enable_fp_fusion=False)
bits=torch.empty((h,n//64,16),device='cuda',dtype=torch.uint32)
_budget_mask[(72,12)](ids,bits,72,count,triton.next_power_of_2(count),num_warps=4)
oe=torch.zeros_like(q,dtype=torch.float32);le=torch.zeros((h,n),device='cuda')
def tail_only():
 return _tail[(72,12)](q,keys,mean,detail,bits,None,None,None,p,n,h,D,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=new,HEAD_PAGES=hp,EXACT=oe,ELSE=le,num_warps=4,num_stages=2,enable_fp_fusion=False)
from local_experiments.review_v1.tuned_attention import _tail
if __name__=='__main__':
 results={};graphs=[timed(base)]
 for name,prefix,skip in variants[1:]:
  launch=lambda:run(prefix,skip)
  ker=launch()
  delta=new.float()-old.float()
  results[name]=dict(bitwise=torch.equal(old.view(torch.uint8),new.view(torch.uint8)),max_abs=float(delta.abs().max()),nmse=float(delta.square().sum()/old.float().square().sum()),registers=ker.n_regs,spills=ker.n_spills,shared=ker.metadata.shared)
  graphs.append(timed(launch))
 graphs.append(timed(tail_only))
 samples=times(graphs)
 for name,ts in zip(['control','skip','prefix','both','tail_only'],samples):
  results.setdefault(name,{}).update(ms=median(ts),speedup=median(a/b for a,b in zip(samples[0],ts)),samples=ts)
 (root/f'module_{p}.json').write_text(json.dumps(results,indent=2))
 print({k:v['ms'] for k,v in results.items()})
