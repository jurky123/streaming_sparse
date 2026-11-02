"""Two self-conditioned denoising steps/case with common random numbers."""
import gc,json,os,signal,time
from pathlib import Path
import torch
from .broad_common import Engine,cases,noise,ROOT as DATA
from .rollout_suite import generate_chunk
from .high_sparse_output_runtime import OutputRuntime,POLICIES
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .improve_accept import Telemetry
ROOT=Path('artifacts/high_sparse_twostep_20261031');MODEL=Path('artifacts/high_sparse_gain_20261031/model.pt')
@torch.no_grad()
def main():
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        files=['high_sparse_twostep_probe.py','high_sparse_output_runtime.py','high_sparse_gain_runtime.py','high_sparse_gain.py']
        identity=dict(code={p:digest(Path('local_experiments')/p) for p in files},model=digest(MODEL),policies=POLICIES,chunks=[2,5],step_rule='2*(case index//2 modulo 2)',source_rule='off/all alternating every 4 cases',train=cases('train'),val=cases('val'),teacher='two self-conditioned dense steps, same initial sparse-history KV and first noisy input, common noise; no full dense-history rollout')
        identity=json.loads(json.dumps(identity))
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        engine=Engine();base_forward=engine.forward
        for split in ['train','val']:
            for idx,case in enumerate(cases(split)):
                dest=ROOT/split/(case['id']+'.json')
                if dest.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted'));return
                disk_guard(ROOT,50,1);chunk_target=[2,5][idx%2];step_target=2*((idx//2)%2);source='all' if idx//4%2 else 'off'
                # Validation uses the trained student on alternating steps.
                if split=='val':source='all' if idx%2 else 'off'
                atomic(ROOT/'status.json',dict(status='collecting',split=split,case=case['id'],step=step_target,source=source,chunk=chunk_target,pid=os.getpid()))
                rt=OutputRuntime(engine,MODEL,source);rows=[];extra_seconds=0.;base_seconds=0.;base_calls=0;expected_next=[None]
                def measured(fn):
                    torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize();return out,time.perf_counter()-t
                def forward(x,emb,chunk,step):
                    nonlocal extra_seconds,base_seconds,base_calls
                    rt.policy_name=source;rt.sparse=True
                    original,dt=measured(lambda:base_forward(x,emb,chunk,step));base_seconds+=dt;base_calls+=1
                    if expected_next[0] is not None and chunk==chunk_target and step==step_target+1:
                        assert torch.equal(original,expected_next[0]),'Branch replay contaminated next main step'
                        expected_next[0]=None
                    if chunk!=chunk_target or step!=step_target:return original
                    n=list(rt.n);write=list(rt.write);commits=list(rt.commits)
                    indices=[(int(c['global_end_index']),int(c['local_end_index'])) for c in engine.cache]
                    shared_noise=noise(case['seed'],chunk,step+1)
                    timestep=torch.full((3,),engine.steps[step+1],device='cuda',dtype=torch.float32)
                    def next_input(clean):
                        return engine.wrapper.scheduler.add_noise(clean.flatten(0,1),shared_noise.flatten(0,1),timestep).unflatten(0,(1,3))
                    first={source:original.clone()};second={};durations={}
                    for name in POLICIES:
                        rt.policy_name=name;rt.sparse=True
                        if name==source:one=original;sec1=dt
                        else:
                            one,sec1=measured(lambda:base_forward(x,emb,chunk,step));extra_seconds+=sec1;first[name]=one.clone()
                        nx=next_input(one)
                        two,sec2=measured(lambda:base_forward(nx,emb,chunk,step+1));extra_seconds+=sec2
                        second[name]=two.clone();durations[name]=sec1+sec2
                    rt.sparse=False
                    teacher1,sec1=measured(lambda:base_forward(x,emb,chunk,step));extra_seconds+=sec1
                    teacher2,sec2=measured(lambda:base_forward(next_input(teacher1),emb,chunk,step+1));extra_seconds+=sec2
                    rt.sparse=True;rt.policy_name=source
                    repeated,audit_sec=measured(lambda:base_forward(x,emb,chunk,step));extra_seconds+=audit_sec
                    assert torch.equal(original,repeated),'Replay changed first source step'
                    assert rt.n==n and rt.write==write and rt.commits==commits
                    assert indices==[(int(c['global_end_index']),int(c['local_end_index'])) for c in engine.cache]
                    expected_next[0]=second[source]
                    metrics={}
                    for name in POLICIES:
                        metrics[name]=dict(seconds=durations[name])
                        for label,out,teacher in [('one',first[name],teacher1),('two',second[name],teacher2)]:
                            assert torch.isfinite(out).all() and torch.isfinite(teacher).all()
                            mse=float((out.float()-teacher.float()).square().mean());norm=float(teacher.float().square().mean())
                            metrics[name][label]=dict(mse=mse,nmse=mse/max(norm,1e-8))
                    rows.append(dict(chunk=chunk,step=step,source=source,metrics=metrics,teacher_seconds=sec1+sec2,replay_seconds=audit_sec,replay_bit_exact=True,no_history_commits=True))
                    return original
                engine.forward=forward;tele=Telemetry();tele.start();start=time.perf_counter()
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                    torch.cuda.synchronize();wall=time.perf_counter()-start
                finally:
                    load=tele.finish();engine.forward=base_forward;rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                assert len(rows)==1 and base_calls==45 and expected_next[0] is None
                result=dict(case=case,rows=rows,base_calls=base_calls,extra_calls=14,base_forward_seconds=base_seconds,extra_forward_seconds=extra_seconds,incremental_forward_cost=extra_seconds/base_seconds,wall_seconds=wall,telemetry=load)
                atomic(dest,result);print(json.dumps(dict(case=case['id'],split=split,extra_fraction=extra_seconds/base_seconds,teacher_seconds=rows[0]['teacher_seconds'],nmse={k:v['two']['nmse'] for k,v in rows[0]['metrics'].items()})),flush=True)
        summary={}
        for split in ['train','val']:
            rows=[json.loads(p.read_text()) for p in (ROOT/split).glob('*.json')]
            summary[split]={name:sum(r['rows'][0]['metrics'][name]['two']['nmse'] for r in rows)/len(rows) for name in POLICIES}
        selected=min(summary['train'],key=summary['train'].get)
        atomic(ROOT/'selection.json',dict(policy=selected,means=summary,selection_scope='train two-step self-conditioned x0 NMSE; validation read-only; not video acceptance'))
        atomic(ROOT/'status.json',dict(status='completed',train_cases=8,val_cases=4))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
