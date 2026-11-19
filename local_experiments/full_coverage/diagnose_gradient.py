"""Replay a failed gradient window without changing formal checkpoints."""
import argparse,copy,json
import numpy as np
import torch
from .common import *
from .train import Bundle,load
from .training_execution import step
from .diagnose_nonfinite import stat


def run(seed=17):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=OUT/f'training/C_s{seed}';rec=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    bundle=Bundle(seed);bundle.load_state_dict(rec['model']);opt=torch.optim.Adam([p for p in bundle.parameters() if p.requires_grad],lr=1e-4);opt.load_state_dict(rec['optimizer'])
    paths=case_files('train');rng=np.random.default_rng(seed);schedule=np.concatenate([rng.permutation(len(paths)) for _ in range(2)]);rng.bit_generator.state=rec['rng']
    fit=torch.load(E24/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long();dest=OUT/'execution_optimization/gradient_failure';dest.mkdir(parents=True,exist_ok=True)
    for u in range(rec['update'],rec['update']+96):
        path=paths[schedule[u]];d=load(path);before=copy.deepcopy(rng.bit_generator.state)
        try:step(bundle,opt,d,qs,B,fit,u,rng,reuse_risk=False,stable=True)
        except BaseException as exc:
            tb=exc.__traceback__;local={}
            while tb:
                if tb.tb_frame.f_code.co_name=='step':local=tb.tb_frame.f_locals
                tb=tb.tb_next
            grads={name:stat(p.grad) for name,p in bundle.named_parameters() if p.grad is not None}
            norm64=torch.stack([p.grad.detach().double().norm() for p in bundle.parameters() if p.grad is not None]).norm()
            report=dict(seed=seed,update=u,event=str(path.relative_to(OUT)),error=repr(exc),gradient_norm64=float(norm64) if torch.isfinite(norm64) else None,
                gradients=grads,all_gradient_elements_finite=all(v['finite']==v['elements'] for v in grads.values()),
                forward={n:stat(local[n]) for n in ('loss','approx','res','sl','g','pg','z','uv','beta') if n in local},numeric_repairs=d.get('numeric_repairs',[]))
            atomic(dest/f's{seed}_diagnosis.json',report)
            atomic(dest/f's{seed}_failure_state.pt',dict(model=bundle.state_dict(),optimizer=opt.state_dict(),update=u,event=str(path.relative_to(OUT)),rng=before,source=rec['source'],execution_impl=rec.get('execution_impl')),True)
            print(json.dumps(report));return
    raise RuntimeError('Failure not reproduced in the saved window')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);a=p.parse_args();run(a.seed)
