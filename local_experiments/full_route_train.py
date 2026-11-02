"""All 29 sparse routing layers with complete conditional attention backward."""
import gc,json,os,signal,time
from pathlib import Path
import torch
from .broad_common import Engine,cases,ROOT as DATA
from .rollout_suite import generate_chunk
from .full_route_runtime import FullRouteRuntime
from .high_sparse_grad_smoke import timed
from .learned_cache.io import atomic,digest,checkpoint,restore,RunLock,disk_guard
from .improve_accept import Telemetry
ROOT=Path('artifacts/full_route_grad_20261031');BASE=Path('artifacts/high_sparse_gain_20261031/model.pt')
def main():
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT/'smoke.json').read_text())['passed']
        assert json.loads(Path('artifacts/route_graph_20261031/acceptance/status.json').read_text())['status']=='complete'
        torch.set_num_threads(4);torch.manual_seed(2131);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        files=['full_route_train.py','full_route_runtime.py','full_route_attention.py','route_graph_runtime.py','high_sparse_gain_runtime.py','high_sparse_gain.py','high_sparse_learn.py','high_sparse_runtime.py','compensation_stable_kernels.py','rollout_runtime.py','broad_common.py']
        identity=dict(code={p:digest(Path('local_experiments')/p) for p in files},base=digest(BASE),epochs=2,lr=.003,trust_radius=.1,train=cases('train'),val=cases('val'),events=dict(chunks=[2,5],steps=[0,3]),objective='final x0 NMSE to same-input dense-self-attention teacher; all 29 sparse layers pair STE and complete Q/K/V/summary backward; no local attention auxiliary loss')
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        original=torch.load(BASE,weights_only=True)['models'];anchor={l:original[l]['swap'].cuda().float().clone() for l in range(1,30)};weights={l:torch.nn.Parameter(a.clone()) for l,a in anchor.items()};opt=torch.optim.Adam(list(weights.values()),lr=.003);done=[]
        saved,_=restore(ROOT)
        if saved:
            assert saved['identity']==identity;done=saved['done']
            with torch.no_grad():
                for l,w in weights.items():w.copy_(saved['models'][l]['swap'].cuda())
            opt.load_state_dict(saved['optimizer'])
        def state():
            models={l:{k:v.cpu().clone() for k,v in m.items()} for l,m in original.items()}
            for l,w in weights.items():models[l]['swap']=w.detach().cpu().clone()
            return dict(models=models,optimizer=opt.state_dict(),done=list(done),identity=identity)
        engine=Engine();engine.wrapper.requires_grad_(False);engine.wrapper.model.gradient_checkpointing=False;base_forward=engine.forward
        for epoch in range(2):
            for case in cases('train'):
                key=f'e{epoch}_{case["id"]}'
                if key in done:continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted',done=len(done)));return
                disk_guard(ROOT,50,1);atomic(ROOT/'status.json',dict(status='training',unit=key,done=len(done),pid=os.getpid()))
                rt=FullRouteRuntime(engine,BASE);rt.route_weights=weights;rt.sw.update(weights);rows=[];base_s=0.;extra_s=0.
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
                        gradients={l:float(w.grad.norm()) for l,w in weights.items()}
                        assert all(w.grad is not None and torch.isfinite(w.grad).all() and w.grad.norm()>0 for w in weights.values())
                        assert all(p.grad is None for p in engine.wrapper.parameters())
                        torch.nn.utils.clip_grad_norm_(list(weights.values()),1.,error_if_nonfinite=True)
                        before={l:w.detach().clone() for l,w in weights.items()};opt.step()
                        with torch.no_grad():
                            for l,w in weights.items():w.copy_(torch.maximum(torch.minimum(w,anchor[l]+.1),anchor[l]-.1))
                    changes={l:float((w.detach()-before[l]).norm()) for l,w in weights.items()}
                    assert all(v>0 for v in changes.values())
                    assert rt.n==n and rt.write==writes and rt.commits==commits
                    rows.append(dict(chunk=chunk,step=step,loss=float(loss.detach()),route_grad_norm=sum(gradients.values()),per_layer_grad_norm=gradients,per_layer_update_norm=changes,update_norm=sum(changes.values()),hard_forward_nmse=fidelity,teacher_seconds=teacher_s,student_seconds=student_s,backward_seconds=back_s,extra_peak_bytes=torch.cuda.max_memory_allocated()-memory_before))
                    extra_s+=teacher_s+student_s+back_s;rt.training_event=False
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
            trained={l:w.detach().clone() for l,w in weights.items()}
            for case in cases('val'):
                key=f'e{epoch}_{case["id"]}';path=ROOT/'validation'/(key+'.json')
                if path.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted_validation',unit=key));return
                atomic(ROOT/'status.json',dict(status='validation',unit=key,pid=os.getpid()));rt=FullRouteRuntime(engine,BASE);rows=[]
                def validate(x,emb,chunk,step):
                    rt.sparse=True
                    with torch.no_grad():
                        for l,w in rt.route_weights.items():w.copy_(anchor[l])
                    initial=base_forward(x,emb,chunk,step)
                    if chunk not in [2,5] or step not in [0,3]:return initial
                    with torch.no_grad():
                        for l,w in rt.route_weights.items():w.copy_(trained[l])
                    student=base_forward(x,emb,chunk,step);rt.sparse=False;teacher=base_forward(x,emb,chunk,step);rt.sparse=True
                    with torch.no_grad():
                        for l,w in rt.route_weights.items():w.copy_(anchor[l])
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
                assert len(rows)==4;atomic(path,dict(rows=rows,source='fixed initial all-layer gain route; matched across epochs'));print(json.dumps(dict(validation=key,ratio=sum(r['ratio'] for r in rows)/4)),flush=True)
        evaluations=[]
        for epoch in range(2):
            rows=[r for p in (ROOT/'validation').glob(f'e{epoch}_*.json') for r in json.loads(p.read_text())['rows']]
            evaluations.append(dict(epoch=epoch,mean_ratio=sum(r['ratio'] for r in rows)/len(rows),initial_nmse=sum(r['initial_nmse'] for r in rows)/len(rows),trained_nmse=sum(r['trained_nmse'] for r in rows)/len(rows)))
        selected=min(evaluations,key=lambda r:r['mean_ratio'])['epoch'];atomic(ROOT/'selection.json',dict(epoch=selected,rows=evaluations,scope='fixed-source final x0 validation only; no video pass'))
        atomic(ROOT/'status.json',dict(status='training_validation_complete',cases=len(done),gradient_updates=4*len(done)))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if 'engine' in locals() and 'base_forward' in locals():engine.forward=base_forward
        lock.close()
if __name__=='__main__':main()
