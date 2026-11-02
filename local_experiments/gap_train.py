"""Matched history-policy two-step training of all gap-model sparse parameters."""
import argparse,gc,json,os,signal,time,torch
from pathlib import Path
from .broad_common import Engine,cases,noise,ROOT as DATA
from .gap_runtime import GapRuntime
from .all_sparse_step_context_v2 import install_step_context
from .rollout_suite import generate_chunk
from .high_sparse_grad_smoke import timed
from .learned_cache.io import atomic,digest,checkpoint,restore,RunLock,disk_guard
ROOT=Path('artifacts/four_gaps_20261031');BASE=Path('artifacts/full_route_grad_20261031/epoch1.pt')
@torch.no_grad()
def noisy_rollout(engine,emb,seed,chunk):
    x=noise(seed,chunk,0)
    for step in range(4):
        clean=engine.forward(x,emb,chunk,step)
        if step<3:x=engine.wrapper.scheduler.add_noise(clean.flatten(0,1),noise(seed,chunk,step+1).flatten(0,1),torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)).unflatten(0,(1,3))
    return clean

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',choices=['shared','matched'],required=True);ap.add_argument('--epochs',type=int,default=2);args=ap.parse_args()
    root=ROOT/args.teacher;lock=RunLock(root);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        for c in [2,5]:assert json.loads((ROOT/f'all_sparse_smoke_chunk{c}.json').read_text())['passed']
        torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
        names=['gap_train.py','gap_runtime.py','gap_model.py','gap_route.py','all_sparse_step_context_v2.py','all_sparse_attention.py','conditional_v_kernels.py']
        identity=dict(code={n:digest(Path('local_experiments')/n) for n in names},base=digest(BASE),teacher=args.teacher,epochs=args.epochs,probes=16,train=cases('train'),events=dict(chunks=[2,5],steps=[0,2]),teacher_policy='same338 hard-grouped retention algorithm and current sparse parameters, independent clean history; no future current-clean commit before student supervision' if args.teacher=='matched' else 'dense on student history',objective='two self-conditioned steps, common noise, final x0 NMSE; all sparse modules, frozen backbone',update='one per case; hard grouping detached; pair STE; no cross-chunk BPTT')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False;base_forward=engine.forward
        rt=GapRuntime(engine,BASE,probes=16);model=rt.model;rt.close();del rt
        groups={3e-5:[],1e-4:[],1e-3:[]}
        for name,p in model.named_parameters():groups[1e-3 if name.endswith('.route') else 3e-5 if name.split('.')[-1] in ['bank','group_v','group_gate'] else 1e-4].append(p)
        opt=torch.optim.Adam([dict(params=v,lr=k) for k,v in groups.items()]);done=[]
        saved,_=restore(root)
        if saved:model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);done=saved['done'];assert saved['identity']==identity
        else:atomic(root/'initial.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},identity=identity),True)
        teacher_engine=None
        if args.teacher=='matched':teacher_engine=Engine();teacher_engine.wrapper.requires_grad_(False);teacher_engine.wrapper.model.gradient_checkpointing=False
        def state():return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),done=list(done),identity=identity)
        for epoch in range(args.epochs):
            for case in cases('train'):
                key=f'e{epoch}_{case["id"]}'
                if key in done:continue
                if stop[0]:atomic(root/'status.json',dict(status='interrupted',done=len(done)));return
                disk_guard(root,50,1);atomic(root/'status.json',dict(status='training',unit=key,done=len(done),pid=os.getpid()))
                rt=GapRuntime(engine,BASE,model=model,probes=16);install_step_context(rt);teacher_rt=None
                if teacher_engine is not None:teacher_rt=GapRuntime(teacher_engine,BASE,model=model,probes=16);teacher_rt.sparse=False
                rows=[];opt.zero_grad(set_to_none=True);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                def forward(x,emb,chunk,step):
                    rt.training_event=False;rt.sparse=True;hard=base_forward(x,emb,chunk,step)
                    if chunk not in [2,5] or step not in [0,2]:return hard
                    eps=noise(case['seed'],chunk,step+1);t=torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)
                    def next_input(clean):return engine.wrapper.scheduler.add_noise(clean.flatten(0,1),eps.flatten(0,1),t).unflatten(0,(1,3))
                    if teacher_rt is not None:
                        assert teacher_rt.n==rt.n and teacher_rt.commits==rt.commits
                        before=list(teacher_rt.commits);one=teacher_engine.forward(x,emb,chunk,step);target=teacher_engine.forward(next_input(one),emb,chunk,step+1)
                        assert teacher_rt.commits==before
                    else:
                        rt.sparse=False;one=base_forward(x,emb,chunk,step);target=base_forward(next_input(one),emb,chunk,step+1);rt.sparse=True
                    rt.training_event=True
                    try:
                        with torch.enable_grad():
                            first,fs=timed(lambda:Engine.forward.__wrapped__(engine,x,emb,chunk,step));first.retain_grad()
                            fidelity=float((first.detach().float()-hard.float()).square().sum()/hard.float().square().sum().clamp_min(1e-8));assert fidelity<1e-7,fidelity
                            pred,sec=timed(lambda:Engine.forward.__wrapped__(engine,next_input(first),emb,chunk,step+1));fs+=sec
                            loss=(pred.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-8);_,bs=timed(lambda:(loss/4).backward())
                        assert first.grad is not None and torch.isfinite(first.grad).all() and first.grad.norm()>0
                        assert rt.isolated_layer0_calls>=2 and not engine.cache[0]['k'].requires_grad
                        rows.append(dict(chunk=chunk,step=step,loss=float(loss.detach()),forward_seconds=fs,backward_seconds=bs,first_step_grad_norm=float(first.grad.norm()),hard_nmse=fidelity,history_policy_matched=teacher_rt is not None))
                        atomic(root/'progress.json',dict(unit=key,events=rows))
                    finally:rt.training_event=False
                    return hard
                engine.forward=forward
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(6):
                        if teacher_engine is not None:clean=noisy_rollout(teacher_engine,emb,case['seed'],c)
                        generate_chunk(engine,emb,case['seed'],c)
                        if teacher_engine is not None:
                            teacher_engine.forward(clean,emb,c,4)
                            assert teacher_rt.n==rt.n and teacher_rt.commits==rt.commits
                    assert len(rows)==4
                    norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
                    assert len(norms)==len(list(model.parameters())) and all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters())
                    assert all(p.grad is None for p in engine.wrapper.parameters())
                    if teacher_engine is not None:assert all(p.grad is None for p in teacher_engine.wrapper.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();model.project()
                    result=dict(rows=rows,gradient_norms=norms,peak_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-start,persistent_student_bytes=rt.persistent_bytes)
                finally:
                    engine.forward=base_forward;rt.close()
                    if teacher_rt is not None:teacher_rt.close()
                    del rt,teacher_rt;gc.collect();torch.cuda.empty_cache()
                done.append(key);checkpoint(root,state());atomic(root/'training'/f'{key}.json',result);print(json.dumps(dict(unit=key,done=len(done),mean_loss=sum(r['loss'] for r in rows)/4,seconds=result['wall_seconds'])),flush=True)
            atomic(root/f'epoch{epoch}.pt',dict(model=state()['model'],identity=identity),True)
        atomic(root/'status.json',dict(status='training_complete_pending_quality',updates=len(done)))
    except Exception as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
