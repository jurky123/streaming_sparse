"""Profile a deployed partial Risk tile on the 432-page event."""
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.risk_partial_mass import _partial
from local_experiments.video_runtime_v1.risk_bound import bound

event=torch.load(Path('/tmp/jzk_route_mass_inputs/p432_c5.pt'),weights_only=True)
q,k,stats,theta=(event[key].cuda().contiguous() for key in ('q','keys','stats','theta'))
h,t=q.shape[:2]
p=stats.shape[1]
u=q.float().square().reshape(h,t,16,8,16).sum(-1)/128
qn=q.float().norm(dim=-1)/(128**.5)
c=bound(stats).contiguous()
st=stats.flatten(1,2).transpose(1,2).contiguous()
v=u.new_empty((h*t,4,16,128))
n=u.new_empty((h*t,4,16,2))
def run():
    return _partial[(h*t,4)](q,k,u,qn,st,theta,c,v,n,t,p,128,True,num_warps=4,enable_fp_fusion=False)
for _ in range(50):
    run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
