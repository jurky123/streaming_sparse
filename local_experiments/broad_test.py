"""Frozen independent test, resumable at each case/chunk with replay audit."""
import argparse
import json
import signal
from pathlib import Path
import torch
from .broad_common import ROOT, CHUNKS, cases, guard, Engine
from .broad_router import LayerRouter, Events, evaluate_event
from .learned_cache.io import atomic, digest, RunLock

METHODS = ['mean', 'reconstruction', 'split']

@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    shards = [ROOT/'train_00_15', ROOT/'train_15_30']
    for p in shards:
        assert json.loads((p/'status.json').read_text())['status'] == 'completed', p
    root = ROOT/f'test_worker{args.worker}'
    lock = RunLock(root)
    stop = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    selected = cases('test')[args.worker::args.workers]
    identity = dict(worker=args.worker, workers=args.workers,
        weights={str(p): digest(p/'best.pt') for p in shards},
        code={str(p):digest(p) for p in [Path(__file__), Path('local_experiments/broad_router.py'), Path('local_experiments/broad_common.py')]},
        dictionary=digest(ROOT/'dictionary.pt'),
        data={str(p):digest(p) for c in selected for p in sorted((ROOT/'trajectories'/c['id']).glob('chunk*.pt'))})
    if (root/'identity.json').exists():
        assert json.loads((root/'identity.json').read_text()) == identity
    else:
        atomic(root/'identity.json', identity)
    models={m:{} for m in METHODS if m!='mean'}
    for p in shards:
        saved=torch.load(p/'best.pt', weights_only=False, map_location='cpu')
        for method in models:
            for layer, weights in saved['models'][method].items():
                model=LayerRouter(layer).cuda().eval()
                model.load_state_dict(weights)
                models[method][layer]=model
    assert all(len(v)==30 for v in models.values())
    engine=Engine();events=Events(range(30));units=0
    try:
        for case in selected:
            if stop[0]:break
            paths={c:root/f'{case["id"]}_c{c:02d}.json' for c in CHUNKS}
            if all(p.exists() for p in paths.values()):continue
            engine.reset();events.reset();engine.callback=None
            emb=torch.load(ROOT/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
            for chunk in range(41):
                data=torch.load(ROOT/'trajectories'/case['id']/f'chunk{chunk:02d}.pt',weights_only=True)
                if chunk in CHUNKS and not paths[chunk].exists():
                    rows=[];audits=[]
                    def callback(current,q,k,v):
                        layer=current['layer'];step=current['step']
                        ids=torch.linspace(0,q.shape[1]-1,4,device='cuda').long()
                        actual=q[0,ids].transpose(0,1).cpu()
                        assert torch.equal(actual,data['queries'][step,layer]), (case['id'],chunk,step,layer,'replay mismatch')
                        audits.append((step,layer))
                        e=events.event(current,q,k,v)
                        for method in METHODS:
                            metrics=evaluate_event(None if method=='mean' else models[method][layer],e)
                            rows.append(dict(case=case['id'],prompt=case['name'],seed=case['seed'],chunk=chunk,
                                step=step,layer=layer,method=method,**metrics))
                    engine.callback=callback
                    for step in range(4):engine.forward(data['noisy'][step].cuda(),emb,chunk,step)
                    engine.callback=None
                    assert len(audits)==120 and len(rows)==360
                    guard();atomic(paths[chunk],dict(rows=rows,replay_exact=True,audited_layer_steps=len(audits)))
                    units+=1
                    atomic(root/'status.json',dict(status='running',case=case['id'],chunk=chunk,disk=guard()))
                    print(case['id'],chunk,'complete; replay exact',flush=True)
                    if args.max_units and units>=args.max_units:stop[0]=True
                    if stop[0]:break
                engine.forward(data['clean'].cuda(),emb,chunk,4)
        complete=all((root/f'{c["id"]}_c{k:02d}.json').exists() for c in selected for k in CHUNKS)
        atomic(root/'status.json',dict(status='completed' if complete else 'interrupted',disk=guard()))
    except Exception as exc:
        atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        engine.callback=None;lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-units',type=int,default=0)
    run(p.parse_args())
