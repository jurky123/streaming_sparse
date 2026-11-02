"""Pairwise fixed-budget improvement, mixed baseline/student case trajectories."""
import gc,json,os,signal
from pathlib import Path
import torch
from .broad_common import Engine,cases,ROOT as DATA
from .high_sparse_learn import Layer
from .high_sparse_gain import inputs,targets,loss,evaluate
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_train import Observer
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,digest,checkpoint,restore,RunLock,disk_guard
ROOT=Path('artifacts/high_sparse_gain_20261031')
def main():
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT/'checks.json').read_text())['passed']
        torch.set_num_threads(4);torch.manual_seed(1931);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        files=['high_sparse_gain_train.py','high_sparse_gain.py','high_sparse_gain_runtime.py','high_sparse_learn.py','high_sparse_learn_kernels.py','high_sparse_train.py']
        identity=dict(code={p:digest(Path('local_experiments')/p) for p in files},initial=digest('artifacts/compensation_probe_20261030/model.pt'),epochs=2,train=cases('train'),val=cases('val'),source='alternating baseline/student by case and epoch; frozen case snapshot; original V fixed',objective='simultaneous 8x8 candidate swaps under compensated normalized attention; weighted pair ranking; bounded score correction')
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        old=torch.load('artifacts/compensation_probe_20261030/model.pt',weights_only=True)['models'];models={l:Layer(old[l]).state_dict() for l in range(1,30)}
        w={l:torch.zeros(12,14,device='cuda',requires_grad=True) for l in models};opts={l:torch.optim.Adam([w[l]],lr=.003) for l in w};done=[]
        saved,_=restore(ROOT)
        if saved is not None:
            assert saved['identity']==identity;done=saved['done']
            for l in w:
                with torch.no_grad():w[l].copy_(saved['models'][l]['swap'].cuda())
                opts[l].load_state_dict(saved['optimizers'][l])
        def pack():
            m={l:{k:v.cpu().clone() for k,v in x.items()} for l,x in models.items()}
            for l in w:m[l]['swap']=w[l].detach().cpu().clone()
            return dict(identity=identity,models=m,done=list(done),optimizers={l:o.state_dict() for l,o in opts.items()})
        engine=Engine()
        for epoch in range(2):
            for idx,case in enumerate(cases('train')):
                key=f'e{epoch}_{case["id"]}';strength=float((epoch+idx)%2)
                if key in done:continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted',done=len(done)));return
                disk_guard(ROOT,50,1);snapshot=ROOT/'source.pt';atomic(snapshot,dict(models=pack()['models']),True)
                atomic(ROOT/'status.json',dict(status='training',unit=key,student=strength,pid=os.getpid(),done=len(done),expected=16))
                rt=GainRuntime(engine,snapshot,strength);rows=[]
                def train(rt,e):
                    l=rt.active_layer;av=rt.centers[l][:,None]+e['old_code']@rt.basis[l].float().transpose(1,2)
                    mask,feat=inputs(e);add,drop,gain,before=targets(e,av,mask)
                    with torch.enable_grad():
                        objective=loss(e,feat,w[l],add,drop,gain,before);opts[l].zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_([w[l]],1.,error_if_nonfinite=True);opts[l].step()
                    with torch.no_grad():w[l].clamp_(-3,3)
                    rows.append(dict(layer=l,chunk=engine.current['chunk'],step=engine.current['step'],loss=float(objective.detach()),positive_pairs=float((gain>0).float().mean()),oracle_best_relative_gain=float((gain.flatten(-2).max(-1).values/before.clamp_min(1e-6)).mean())))
                rt.kernel=Observer(rt,rt.kernel,train)
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                assert len(rows)==174;done.append(key);checkpoint(ROOT,pack());atomic(ROOT/'training'/(key+'.json'),dict(source_strength=strength,rows=rows))
                print(json.dumps(dict(unit=key,done=len(done),loss=sum(r['loss'] for r in rows)/len(rows),student=strength)),flush=True)
        final=ROOT/'model.pt'
        if not final.exists():atomic(final,dict(models=pack()['models'],identity=identity),True)
        for source in [0.,1.]:
            for case in cases('val'):
                key=f'source{source}_{case["id"]}';path=ROOT/'validation'/(key+'.json')
                if path.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted_validation',unit=key));return
                disk_guard(ROOT,50,1);atomic(ROOT/'status.json',dict(status='validation',unit=key,pid=os.getpid()))
                rt=GainRuntime(engine,final,source);rows=[]
                def validation(rt,e):
                    l=rt.active_layer;av=rt.centers[l][:,None]+e['old_code']@rt.basis[l].float().transpose(1,2)
                    rows.append(dict(layer=l,chunk=engine.current['chunk'],step=engine.current['step'],metrics=evaluate(e,av,w[l])))
                rt.kernel=Observer(rt,rt.kernel,validation)
                try:
                    emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                    for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                assert len(rows)==174;atomic(path,dict(source=source,case=case['name'],rows=rows));print(json.dumps(dict(validation=key)),flush=True)
        summary={}
        for source in [0.,1.]:
            rows=[r for p in (ROOT/'validation').glob(f'source{source}_*.json') for r in json.loads(p.read_text())['rows']]
            summary[str(source)]={s:sum(r['metrics'][s]['nmse'] for r in rows)/len(rows) for s in ['0.0','0.25','0.5','1.0']}
        eligible=[s for s in ['0.0','0.25','0.5','1.0'] if all(m[s]<=m['0.0']*1.02 for m in summary.values())]
        best=min(eligible,key=lambda s:sum(m[s]/m['0.0'] for m in summary.values()))
        atomic(ROOT/'selection.json',dict(by_source=summary,strength=float(best),scope='local validation on baseline and student trajectories only; no video acceptance; strengths predeclared'))
        atomic(ROOT/'status.json',dict(status='training_validation_complete',units=len(done)))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
