"""Isolate shared QK and tail GEMMs on the same city_walk capture."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .benchmark import measure
from .tensor_bmm import tensor_bmm

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rt,e,identity=inputs(True);q=e['q'].flatten(1,2);s=e['s'];h,m,d=q.shape;p=s.mean.shape[1]
    # Actual probabilities from the captured query/summary, without random fixtures.
    z=torch.bmm(q,s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,m,p,4)/d**.5
    w=z.flatten(-2).softmax(-1).reshape_as(z)
    from ..review_v1.ops import haar
    contrast=torch.einsum('hmpr,ar->hmpa',w,haar(q)).reshape(h,m,p*3)
    pairs=dict(qk=(q,s.keys.float().flatten(1,2).transpose(1,2)),
               mean=(w.sum(-1),s.mean.float()),detail=(contrast,s.detail.float().flatten(1,2)))
    rows={}
    for name,(a,b) in pairs.items():
        old,x=capture(lambda:torch.bmm(a,b));new,y=capture(lambda:tensor_bmm(a,b))
        delta=(x-y).double();err=dict(nmse=float(delta.square().sum()/x.double().square().sum()),max_abs=float(delta.abs().max()),equal=bool(torch.equal(x,y)))
        before=measure(old.replay,25);after=measure(new.replay,25);flops=2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
        rows[name]=dict(**err,before=before,after=after,speedup=before['gpu_ms_median']/after['gpu_ms_median'],gflop=flops/1e9,effective_tflops=flops/after['gpu_ms_median']/1e9)
    path=Path('artifacts/video_runtime_v1/hotspot_round5');path.mkdir(parents=True,exist_ok=True)
    (path/'tensor_bmm.json').write_text(json.dumps(dict(rows=rows,identity=identity),indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':run()
