"""Case-resumable final-x0 STE optimization of layer29 routing coefficients."""
import gc,json,os,signal,time
from pathlib import Path
import torch
from .broad_common import Engine,cases,ROOT as DATA
from .rollout_suite import generate_chunk
from .high_sparse_grad_runtime import GradRuntime
from .high_sparse_grad_smoke import timed
from .learned_cache.io import atomic,digest,checkpoint,restore,RunLock,disk_guard
from .improve_accept import Telemetry
ROOT=Path('artifacts/high_sparse_grad_20261031');BASE=Path('artifacts/high_sparse_gain_20261031/model.pt')
def main():
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT/'smoke.json').read_text())['passed']
        torch.set_num_threads(4);torch.manual_seed(2131);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        files=['high_sparse_grad_train.py','high_sparse_grad_runtime.py','high_sparse_grad_smoke.py','high_sparse_gain_runtime.py','high_sparse_gain.py']
        identity=dict(code={p:digest(Path('local_experiments')/p) for p in files},base=digest(BASE),epochs=2,lr=.01,trust_radius=.3,train=cases('train'),val=cases('val'),events=dict(chunks=[2,5],steps=[0,3]),objective='final x0 NMSE to same-input dense-self-attention teacher; last-layer pair STE only; no local attention auxiliary loss')
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        original=torch.load(BASE,weights_only=True)['models'];anchor=original[29]['swap'].cuda().float().clone();weight=torch.nn.Parameter(anchor.clone());opt=torch.optim.Adam([weight],lr=.01);done=[]
        saved,_=restore(ROOT)
        if saved:
            assert saved['identity']==identity;done=saved['done']
            with torch.no_grad():weight.copy_(saved['models'][29]['swap'].cuda())
            opt.load_state_dict(saved['optimizer'])
        def state():
            models={l:{k:v.cpu().clone() for k,v in m.items()} for l,m in original.items()};models[29]['swap']=weight.detach().cpu().clone()
            return dict(models=models,optimizer=opt.state_dict(),done=list(done),identity=identity)
        engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False;base_forward=engine.forward
        for epoch in range(2):
            for case in cases('train'):
                key=f'e{epoch}_{case["id"]}'
                if key in done:continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted',done=len(done)));return
                disk_guard(ROOT,50,1);atomic(ROOT/'status.json',dict(status='training',unit=key,done=len(done),pid=os.getpid()))
                rt=GradRuntime(engine,BASE);rt.weight=weight;rt.sw[29]=weight;rows=[];base_s=0.;extra_s=0.
                def forward(x,emb,chunk,step):
                    nonlocal base_s,extra_s
                    rt.training_event=False;rt.sparse=True
                    hard,sec=timed(lambda:base_forward(x,emb,chunk,step));base_s+=sec
                    if chunk not in [2,5] or step not in [0,3]:return hard
                    n=list(rt.n);writes=list(rt.write);commits=list(rt.commits)
                    rt.sparse=False;teacher,teacher_s=timed(lambda:base_forward(x,emb,chunk,step));rt.sparse=True;rt.training_event=True
                    memory_before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    with torch.enable_grad():
                        pred,student_s=timed(lambda:Engine.forward.__wrapped__(engine,x,emb,chunk,step))
                        fidelity=float((pred.detach().float()-hard.float()).square().sum()/hard.float().square().sum().clamp_min(1e-8));assert fidelity<1e-7,fidelity
                        norm=teacher.float().square().mean().clamp_min(1e-8);loss=(pred.float()-teacher.float()).square().mean()/norm
                        assert loss.requires_grad
                        opt.zero_grad();_,back_s=timed(loss.backward)
                        grad=float(weight.grad.norm());assert torch.isfinite(weight.grad).all() and grad>0
                        margin_grad=float(rt.last_margin.grad.norm());assert margin_grad>0
                        assert all(p.grad is None for p in engine.wrapper.parameters())
                        torch.nn.utils.clip_grad_norm_([weight],1.,error_if_nonfinite=True);before=weight.detach().clone();opt.step()
                        with torch.no_grad():weight.copy_(torch.maximum(torch.minimum(weight,anchor+.3),anchor-.3))
                    change=float((weight.detach()-before).norm());assert change>0
                    assert rt.n==n and rt.write==writes and rt.commits==commits
                    rows.append(dict(chunk=chunk,step=step,loss=float(loss.detach()),route_grad_norm=grad,margin_grad_norm=margin_grad,update_norm=change,hard_forward_nmse=fidelity,teacher_seconds=teacher_s,student_seconds=student_s,backward_seconds=back_s,extra_peak_bytes=torch.cuda.max_memory_allocated()-memory_before))
                    extra_s+=teacher_s+student_s+back_s;rt.training_event=False;rt.clear_proxy()
                    return hard
                engine.forward=forward;tele=Telemetry();tele.start()
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                finally:load=tele.finish();engine.forward=base_forward;rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                assert len(rows)==4;done.append(key);checkpoint(ROOT,state());atomic(ROOT/'training'/(key+'.json'),dict(rows=rows,base_forward_seconds=base_s,extra_training_seconds=extra_s,extra_fraction=extra_s/base_s,telemetry=load))
                print(json.dumps(dict(unit=key,done=len(done),loss=sum(r['loss'] for r in rows)/4,grad=sum(r['route_grad_norm'] for r in rows)/4,extra_fraction=extra_s/base_s)),flush=True)
            path=ROOT/f'epoch{epoch}.pt'
            if not path.exists():atomic(path,dict(models=state()['models'],epoch=epoch,identity=identity),True)
            trained=weight.detach().clone()
            for case in cases('val'):
                key=f'e{epoch}_{case["id"]}';path=ROOT/'validation'/(key+'.json')
                if path.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted_validation',unit=key));return
                atomic(ROOT/'status.json',dict(status='validation',unit=key,pid=os.getpid()));rt=GradRuntime(engine,BASE);rows=[]
                def validate(x,emb,chunk,step):
                    rt.sparse=True
                    with torch.no_grad():rt.weight.copy_(anchor)
                    initial=base_forward(x,emb,chunk,step)
                    if chunk not in [2,5] or step not in [0,3]:return initial
                    with torch.no_grad():rt.weight.copy_(trained)
                    student=base_forward(x,emb,chunk,step);rt.sparse=False;teacher=base_forward(x,emb,chunk,step);rt.sparse=True
                    with torch.no_grad():rt.weight.copy_(anchor)
                    repeated=base_forward(x,emb,chunk,step);assert torch.equal(initial,repeated)
                    norm=teacher.float().square().mean().clamp_min(1e-8)
                    a=float((initial.float()-teacher.float()).square().mean()/norm);b=float((student.float()-teacher.float()).square().mean()/norm)
                    rows.append(dict(chunk=chunk,step=step,initial_nmse=a,trained_nmse=b,ratio=b/max(a,1e-12)))
                    return initial
                engine.forward=validate
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                finally:engine.forward=base_forward;rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                assert len(rows)==4;atomic(path,dict(rows=rows,source='fixed initial last-layer-only route; matched across epochs'));print(json.dumps(dict(validation=key,ratio=sum(r['ratio'] for r in rows)/4)),flush=True)
        evaluations=[]
        for epoch in range(2):
            rows=[r for p in (ROOT/'validation').glob(f'e{epoch}_*.json') for r in json.loads(p.read_text())['rows']]
            evaluations.append(dict(epoch=epoch,mean_ratio=sum(r['ratio'] for r in rows)/len(rows),initial_nmse=sum(r['initial_nmse'] for r in rows)/len(rows),trained_nmse=sum(r['trained_nmse'] for r in rows)/len(rows)))
        selected=min(evaluations,key=lambda r:r['mean_ratio'])['epoch'];atomic(ROOT/'selection.json',dict(epoch=selected,rows=evaluations,scope='fixed-source final x0 validation only; no video pass'))
        atomic(ROOT/'status.json',dict(status='training_validation_complete',cases=len(done),gradient_updates=4*len(done)))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:engine.forward=base_forward if 'base_forward' in locals() else None;lock.close()
if __name__=='__main__':main()
