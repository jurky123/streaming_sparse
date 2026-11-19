"""Replay original event arithmetic from last good checkpoint; no formal updates."""
import argparse,copy,json,traceback
import numpy as np
import torch
from .common import *
from .train import Bundle,load
from .training_execution import step

def stat(x):
    if not torch.is_tensor(x):return str(type(x))
    y=x.detach();finite=torch.isfinite(y);vals=y[finite]
    return dict(shape=list(y.shape),dtype=str(y.dtype),finite=int(finite.sum()),elements=y.numel(),min=float(vals.min()) if vals.numel() else None,max=float(vals.max()) if vals.numel() else None,zeros=int((y==0).sum()))

def run(seed):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=OUT/f'training/C_s{seed}';rec=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    b=Bundle(seed);b.load_state_dict(rec['model']);opt=torch.optim.Adam([p for p in b.parameters() if p.requires_grad],lr=1e-4);opt.load_state_dict(rec['optimizer'])
    paths=case_files('train');rng=np.random.default_rng(seed);schedule=np.concatenate([rng.permutation(len(paths)) for _ in range(2)]);rng.bit_generator.state=rec['rng']
    fit=torch.load(E24/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long();dest=OUT/'execution_optimization/nonfinite';dest.mkdir(parents=True,exist_ok=True)
    for u in range(rec['update'],rec['update']+96):
        path=paths[schedule[u]];d=load(path);before=copy.deepcopy(rng.bit_generator.state)
        try:row,_,_=step(b,opt,d,qs,B,fit,u,rng,reuse_risk=False)
        except BaseException as exc:
            tb=exc.__traceback__
            while tb.tb_next:tb=tb.tb_next
            local=tb.tb_frame.f_locals;names=('approx','res','sl','loss','z','uv','beta','zt','ut','bt','g','pg','target','mask','r','a')
            report=dict(seed=seed,update=u,event=str(path.relative_to(OUT)),original_arithmetic_reproduces_failure=True,error=repr(exc),variables={n:stat(local[n]) for n in names if n in local},contributions={n:stat(v) for n,v in local.get('c',{}).items()},teacher={n:stat(d[n]) for n in ('N','Z','dense','log_scale','logZ')})
            atomic(dest/f's{seed}_diagnosis.json',report)
            atomic(dest/f's{seed}_failure_state.pt',dict(model=b.state_dict(),optimizer=opt.state_dict(),update=u,event=str(path.relative_to(OUT)),rng=before,source=rec['source']),True)
            print(json.dumps(report));return
    raise RuntimeError('Original path did not reproduce failure in this 96-step window')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);a=p.parse_args();run(a.seed)
