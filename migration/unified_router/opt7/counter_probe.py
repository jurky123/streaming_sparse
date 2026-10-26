import torch,json
from pathlib import Path
from migration.unified_router.opt7.counters import counters
from migration.unified_router.opt5.probe import capture,timing
from migration.unified_router.common import atomic_json
root=Path('streaming_sparse/v10/experiments/full29_keep15_e2e_20261024');data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
mask=data['mask'];cost=data['sketch'].costs
h,n,b=mask.shape;kh=data['k'].shape[1]-4680
@torch.no_grad()
def old():
 count=mask*cost[None,None];rh=count[...,:b-74].sum();rc=count[...,b-74:].sum();nq=h*n
 return torch.stack((rh+rc,rh.new_tensor(nq),rh,rc,rh.new_tensor(nq*(kh+4680)),rh.new_tensor(1), (rh+rc)/(nq*(kh+4680)),rh/(nq*kh),rc/(nq*4680),rh.new_tensor(0)))
y=counters(mask,cost,kh,4680);ref=old();assert torch.equal(y[:6],ref[:6]),(y,ref)
import time,statistics
def measured(fn):
 for _ in range(10):fn()
 values=[]
 for _ in range(5):
  torch.cuda.synchronize();start=time.perf_counter()
  for i in range(100):fn()
  torch.cuda.synchronize();values.append((time.perf_counter()-start)*10)
 return statistics.median(values)
r=dict(before_ms=measured(old),after_ms=measured(lambda:counters(mask,cost,kh,4680)),counts_identical=torch.equal(y[:6],ref[:6]),ratio_max_error=float((y[6:9]-ref[6:9]).abs().max()),avoided_float_tensor_bytes=mask.numel()*4,measurement='Host wall time, 100 calls per sample, synchronization at boundaries; old CPU scalar construction is not CUDA-graph capturable')
print(r);atomic_json(root/'optimization7/counters_probe.json',r)
