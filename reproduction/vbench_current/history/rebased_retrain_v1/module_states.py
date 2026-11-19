import json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from policy import initial,pairs,state
from scan import counts
from local_experiments.full_coverage import training_execution as ex,train_full_pset as ps
from local_experiments.full_coverage.stable_targets import true_gains
from local_experiments.review_v1.ops import RouteOps
from local_experiments.e24_joint_representatives.core import summary,contributions

@torch.no_grad()
def run():
    root=Path(__file__).parent
    torch.set_num_threads(2)
    b,_,_,_,_=ps.models(17)
    qs=torch.linspace(0,63,16,device='cuda').round().long()
    times=[];start=time.monotonic()
    for path in sorted((root.parent/'rebased_budget_ablation_v1/train_pilot').glob('*_teacher.pt')):
        d=ex.to_device(ex.load_cpu(path));m=b.layers[d['key']]
        s=summary(m,d['k'],d['v'],d['hp']);c=contributions(d['q'],s,m.basis,d['log_scale'])
        f,_=RouteOps.features(d['q'][qs][None],s)
        risk=(f*m.theta[:,None,None]).sum(-1).mean(1)[0]
        own=torch.as_tensor(d['own'],device=risk.device)
        for idx in (11,21):
            quota=counts(len(risk))[idx];mask=initial(risk,own,quota);r,a=pairs(risk,mask,own)
            for mode in (0,6,7,8,9):
                t=time.monotonic(); mm,rr,aa=state(c,d,s,m,mask,r,a,mode,risk,np.random.default_rng(701+mode))
                assert mm[:24].all() and mm[own] and mm[-72:].sum()==quota[0] and mm[24:-72].sum()==quota[1]
                check=pairs(risk,mm,own)
                assert len(rr)>0 and torch.equal(rr,check[0]) and torch.equal(aa,check[1])
                g=true_gains(c,d,s,m,mm,rr,aa);assert g.isfinite().all()
                times.append(dict(chunk=d['ev']['chunk'],index=idx,mode=mode,pairs=len(g),swapped=int((mm!=mask).sum()/2),seconds=time.monotonic()-t))
    (root/'module_states.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,states=times),indent=2))
run()
