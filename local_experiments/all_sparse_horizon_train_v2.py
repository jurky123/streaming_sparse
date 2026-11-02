"""Case-boundary updates and atomic resume; frozen backbone, all sparse modules."""
import argparse,gc,json,os,signal
from pathlib import Path
import torch
from .broad_common import Engine,cases,noise,ROOT as DATA
from .all_sparse_step_context_v2 import install_step_context
from .rollout_suite import generate_chunk
from .all_sparse_runtime import AllSparseRuntime
from .high_sparse_grad_smoke import timed
from .learned_cache.io import atomic,digest,checkpoint,restore,RunLock,disk_guard
INITIAL=Path('artifacts/sparse_expansion_20261031/all_sparse_train/initial.pt')
BASE=Path('artifacts/full_route_grad_20261031/epoch1.pt')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--horizon',type=int,choices=[1,2],required=True);args=ap.parse_args()
    ROOT=Path(f'artifacts/sparse_expansion_20261031/horizon{args.horizon}_train')
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT.parent/'twostep_backward_checks.json').read_text())['passed']
        for chunk in [2,5]:assert json.loads((ROOT.parent/f'all_sparse_smoke_chunk{chunk}.json').read_text())['passed']
        torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
        files=['all_sparse_horizon_train_v2.py','all_sparse_step_context_v2.py','all_sparse_model.py','all_sparse_runtime.py','all_sparse_attention.py','conditional_v_kernels.py']
        identity=dict(code={p:digest(Path('local_experiments')/p) for p in files},base=digest(BASE),initial=digest(INITIAL),horizon=args.horizon,epochs=1,probes=8,swaps=1,train=cases('train'),events=dict(chunks=[2,5],steps=[0,2]),update='once per case; four equally weighted final-x0 NMSE losses after chosen horizon; matched optimizer/events/initialization; no cross-chunk BPTT',backbone='frozen',grouping='hard physical pages; pooling feature path differentiable; sorting detached')
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False;base_forward=engine.forward
        rt=AllSparseRuntime(engine,BASE);model=rt.model;model.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['model']);rt.close();del rt
        groups={3e-5:[],1e-4:[],1e-3:[]}
        for name,p in model.named_parameters():groups[1e-3 if name.endswith('.route') else 3e-5 if name.split('.')[-1] in ['bank','group_v','group_gate'] else 1e-4].append(p)
        opt=torch.optim.Adam([dict(params=v,lr=k) for k,v in groups.items()]);done=[]
        saved,_=restore(ROOT)
        if saved:model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);done=saved['done'];assert saved['identity']==identity
        elif not (ROOT/'initial.pt').exists():atomic(ROOT/'initial.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},identity=identity),True)
        def state():return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),done=list(done),identity=identity)
        for epoch in range(1):
            for case in cases('train'):
                key=f'e{epoch}_{case["id"]}'
                if key in done:continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted',done=len(done)));return
                disk_guard(ROOT,50,1);atomic(ROOT/'status.json',dict(status='training',unit=key,done=len(done),pid=os.getpid()))
                rt=AllSparseRuntime(engine,BASE,model=model);install_step_context(rt);rows=[];opt.zero_grad(set_to_none=True)
                def forward(x,emb,chunk,step):
                    rt.training_event=False;rt.sparse=True
                    hard=base_forward(x,emb,chunk,step)
                    if chunk not in [2,5] or step not in [0,2]:return hard
                    eps=noise(case['seed'],chunk,step+1);t=torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)
                    def next_input(clean):return engine.wrapper.scheduler.add_noise(clean.flatten(0,1),eps.flatten(0,1),t).unflatten(0,(1,3))
                    rt.sparse=False;teacher=base_forward(x,emb,chunk,step)
                    if args.horizon==2:teacher=base_forward(next_input(teacher),emb,chunk,step+1)
                    rt.sparse=True;rt.training_event=True
                    try:
                        with torch.enable_grad():
                            first,fs=timed(lambda:Engine.forward.__wrapped__(engine,x,emb,chunk,step))
                            error=float((first.detach().float()-hard.float()).square().sum()/hard.float().square().sum().clamp_min(1e-8));assert error<1e-7,error
                            pred=first
                            if args.horizon==2:
                                first.retain_grad();pred,sec=timed(lambda:Engine.forward.__wrapped__(engine,next_input(first),emb,chunk,step+1));fs+=sec
                            loss=(pred.float()-teacher.float()).square().mean()/teacher.float().square().mean().clamp_min(1e-8)
                            _,bs=timed(lambda:(loss/4).backward())
                        first_grad=None if args.horizon==1 else float(first.grad.norm())
                        if args.horizon==2:
                            assert torch.isfinite(first.grad).all() and first_grad>0
                            assert rt.isolated_layer0_calls>=2
                            assert not engine.cache[0]['k'].requires_grad and not engine.cache[0]['v'].requires_grad
                        rows.append(dict(chunk=chunk,step=step,horizon=args.horizon,loss=float(loss.detach()),forward_seconds=fs,backward_seconds=bs,hard_nmse=error,first_step_gradient_norm=first_grad))
                        atomic(ROOT/'progress.json',dict(unit=key,events=rows))
                    finally:rt.training_event=False
                    return hard
                engine.forward=forward
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(6):generate_chunk(engine,emb,case['seed'],c)
                    assert len(rows)==4
                    norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
                    assert len(norms)==len(list(model.parameters())) and all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters())
                    assert all(p.grad is None and not p.requires_grad for p in engine.wrapper.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();model.project()
                finally:engine.forward=base_forward;rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                done.append(key);checkpoint(ROOT,state());atomic(ROOT/'training'/f'{key}.json',dict(rows=rows,gradient_norms=norms))
                print(json.dumps(dict(unit=key,done=len(done),mean_loss=sum(r['loss'] for r in rows)/4)),flush=True)
            atomic(ROOT/f'epoch{epoch}.pt',dict(model=state()['model'],identity=identity),True)
        atomic(ROOT/'status.json',dict(status='pilot_training_complete_pending_validation',done=len(done),updates=len(done)))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
