import json,time,statistics
from pathlib import Path
import torch
from local_experiments.full_coverage import training_execution as ex,train_full_pset as ps
from local_experiments.e24_joint_representatives.core import summary,contributions
from local_experiments.review_v1.ops import RouteOps
from local_experiments.e30_beta_diagnostics.run import local_scores
from policy import initial,pairs
from scan import counts
from local_experiments.full_coverage.stable_targets import true_gains
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False

@torch.no_grad()
def run():
    root=Path(__file__).parent
    b,phi,reader,fit,_=ps.models(17)
    reader.load_state_dict(torch.load(ps.OUT/'training/Full_Pset_s17/selected.pt',weights_only=False)['model'])
    roof=json.loads((root.parent/'rebased_budget_ablation_v1/roof.json').read_text())
    qs=torch.linspace(0,63,16,device='cuda').round().long()
    rows=[]
    for chunk in (2,6):
        path=next((root/'records/train/bird_flight_1201').glob(f'c{chunk}_l5_h4_*_teacher.pt'))
        d=ex.to_device(ex.load_cpu(path));m=b.layers[d['key']];p=d['k'].shape[1]
        def bench(name,fn,work,prec,scope):
            out=fn(); samples=[]
            for _ in range(5):
                a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(5):out=fn()
                z.record();z.synchronize();samples.append(a.elapsed_time(z)/5)
            ms=statistics.median(samples)
            rate=roof['torch.'+prec]['tflops'] if prec else None
            ref=work/(rate*1e9) if rate else None
            rows.append(dict(pages=p,module=name,measured_ms=ms,known_flops=work,reference_tflops=rate,reference_ms=ref,measured_minus_reference_ms=(ms-ref if ref else None),scope=scope))
            return out
        s=bench('summary',lambda:summary(m,d['k'],d['v'],d['hp']),2*p*64*128*32,'float32','one K feature projection only; history V and other operations omitted')
        c=bench('contributions',lambda:contributions(d['q'],s,m.basis,d['log_scale']),2*2*64*4*p*128,'float64','representative QK and weighted representative value; detail reconstruction and pointwise operations omitted')
        f,_=bench('risk_features',lambda:RouteOps.features(d['q'][qs][None],s),2*2*16*p*4*128,'float32','two representative QK passes')
        risk=(f*m.theta[:,None,None]).sum(-1).mean(1)[0]; own=torch.tensor(d['own'],device='cuda')
        mask=initial(risk,own,counts(p)[11]); r,a=pairs(risk,mask,own)
        bench('teacher_gain',lambda:true_gains(c,d,s,m,mask,r,a),None,None,'irregular gathers, reductions; no homogeneous rate')
        F=phi(d['q'][qs],s)
        bench('LocalReader',lambda:reader(s,F),2*p*(712*64+64*85+16*4*17),'float32','MLP and response contraction only')
    (root/'training_module_profile.json').write_text(json.dumps(rows,indent=2))
run()
