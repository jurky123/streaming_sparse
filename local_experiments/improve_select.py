"""Frozen selected weights are re-evaluated on validation, never confirmation videos."""
import json
from pathlib import Path
import torch
from .broad_common import Engine,cases,CHUNKS,ROOT as DATA
from .improve_models import Model,Events,METHODS,objective,evaluate,swaps
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/rollout_improve_20261030')
@torch.no_grad()
def main():
    root=ROOT/'selected_val';lock=RunLock(root)
    try:
        assert json.loads((ROOT/'pilot/status.json').read_text())['status']=='completed'
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        saved=torch.load(ROOT/'pilot/best.pt',weights_only=False,map_location='cpu');layers=saved['identity']['layers']
        identity=dict(weights=digest(ROOT/'pilot/best.pt'),code={f:digest(Path('local_experiments')/f) for f in ['improve_models.py','improve_select.py']})
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        models={m:{} for m in METHODS}
        for m in METHODS:
            for l in layers:
                model=Model(l,m).cuda().eval();model.load_state_dict(saved['models'][m][l]);models[m][l]=model
        engine=Engine();events=Events(layers)
        for case in cases('val'):
            paths={c:root/f'{case["id"]}.c{c}.json' for c in CHUNKS}
            if all(p.exists() for p in paths.values()):continue
            emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda();engine.reset();events.reset();engine.callback=None
            for chunk in range(41):
                data=torch.load(DATA/'trajectories'/case['id']/f'chunk{chunk:02d}.pt',weights_only=True)
                if chunk in CHUNKS and not paths[chunk].exists():
                    rows=[]
                    def callback(current,q,k,v):
                        l=current['layer']
                        if l not in events.layers:return
                        ids=torch.linspace(0,4607,4,device=q.device).long()
                        assert torch.equal(q[0,ids].transpose(0,1).cpu(),data['queries'][current['step'],l]),'replay mismatch'
                        e=events.event(current,q,k,v)
                        for m in METHODS:
                            result,z,mask=evaluate(models[m][l],e)
                            rows.append(dict(case=case['name'],layer=l,chunk=chunk,step=current['step'],method=m,**result,oracle=swaps(e,z,mask),proxy=swaps(e,z,mask,True)))
                    engine.callback=callback
                    for step in range(4):engine.forward(data['noisy'][step].cuda(),emb,chunk,step)
                    engine.callback=None;assert len(rows)==len(layers)*4*len(METHODS)
                    atomic(paths[chunk],dict(rows=rows,replay_exact=True));print(case['name'],chunk,flush=True)
                    disk_guard(ROOT,50,4)
                engine.forward(data['clean'].cuda(),emb,chunk,4)
        rows=[r for p in root.glob('*.c*.json') for r in json.loads(p.read_text())['rows']]
        def average(m,key='nmse',case=None):
            data=[r for r in rows if r['method']==m and (case is None or r['case']==case)]
            vals=[v for r in data for v in (r[key]['nmse'] if key in ['oracle','proxy'] else r[key])]
            return sum(vals)/len(vals)
        scores={m:dict(nmse=average(m),oracle=average(m,'oracle'),proxy=average(m,'proxy'),max_query_mean=average(m,'max_query_nmse')) for m in METHODS}
        winner=min(METHODS[1:],key=lambda m:scores[m]['nmse'])
        proxy_good=scores[winner]['proxy']<scores[winner]['nmse']*.995 and all(average(winner,'proxy',c['name'])<=average(winner,case=c['name'])*1.02 for c in cases('val'))
        result=dict(selected_method=winner,layers=layers,proxy_promoted=proxy_good,offline_gate_passed=scores[winner]['nmse']<scores['anchor']['nmse']*.995,
            scores=scores,weights_sha256=identity['weights'],scope='6-layer pilot; other23 sparse layers retain frozen mass_var weights; layer0 dense',
            acceptance_cases=['basketball_court','butterfly_garden','harbor_cranes','lantern_festival'],acceptance_seeds=[808,909],
            note='Best mechanism enters actual rollout even if offline gate fails, to report a negative result honestly. No test-seed tuning.')
        atomic(ROOT/'selection.json',result);atomic(root/'status.json',dict(completed=True));print(json.dumps(result),flush=True)
    except Exception as exc:atomic(root/'status.json',dict(completed=False,error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
