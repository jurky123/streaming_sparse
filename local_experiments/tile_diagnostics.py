"""Validation-only diagnostic: exact route mass versus shared multi-query mass."""
import json,torch
from .tile_common import ROOT,cases,CHUNKS,Engine,guard
from .tile_router import Events
from .learned_cache.data import teacher
from .learned_cache.model import initial_mask,predict,output_for_masks
from .learned_cache.run import nmse
from .learned_cache.io import atomic,RunLock,digest
LAYERS=[0,4,14,17,24,29]

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'val_diagnostics';lock=RunLock(root)
    identity=dict(code=digest(__file__),layers=LAYERS,cases=cases('val'),scope='validation only; diagnostic scorers are not output-optimal oracles')
    if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
    else:atomic(root/'identity.json',identity)
    engine=Engine();events=Events(LAYERS);events.training=True
    try:
        for case in cases('val'):
            if all((root/f'{case["id"]}_c{ch:02d}.json').exists() for ch in CHUNKS):continue
            events.reset();engine.reset();engine.callback=None
            emb=torch.load(ROOT/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
            for chunk in range(41):
                data=torch.load(ROOT/'trajectories'/case['id']/f'chunk{chunk:02d}.pt',weights_only=True)
                path=root/f'{case["id"]}_c{chunk:02d}.json'
                if chunk in CHUNKS and not path.exists():
                    rows=[]
                    def callback(current,q,k,v):
                        if current['layer'] not in events.layers:return
                        e=events.event(current,q,k,v)
                        rz,_,_=teacher(e['route'][:,::4],e['h']['k'],e['h']['v'],e['kc'],e['vc'])
                        actual_mass=e['z'].softmax(-1).reshape(12,8,4,361)
                        shared_mass=actual_mass.mean(2)
                        mz,_,_=predict(e['route'],e['h']['mean'],e['kc'],e['vc'])
                        masks={'mean_proto':initial_mask(mz),
                            'exact_tile_query':initial_mask(rz).repeat_interleave(4,1),
                            'shared_actual_mass':initial_mask(shared_mass.clamp_min(1e-30).log()).repeat_interleave(4,1),
                            'per_query_mass':initial_mask(e['z'])}
                        scores={m:nmse(output_for_masks(e['z'],e['values'],mask[:,:,None])[:,:,0],e['target']).mean(-1).cpu().tolist() for m,mask in masks.items()}
                        change=(masks['exact_tile_query'][...,:-1]!=masks['shared_actual_mass'][...,:-1]).sum(-1).float()/144
                        ratio=e['route'].norm(dim=-1)/e['q'].norm(dim=-1).clamp_min(1e-8)
                        rows.append(dict(layer=current['layer'],step=current['step'],case=case['id'],chunk=chunk,
                            nmse=scores,replaced_fraction=change.mean(-1).cpu().tolist(),route_norm_ratio=ratio.mean(-1).cpu().tolist()))
                    engine.callback=callback
                    for step in range(4):engine.forward(data['noisy'][step].cuda(),emb,chunk,step)
                    engine.callback=None;assert len(rows)==24;guard();atomic(path,rows)
                engine.forward(data['clean'].cuda(),emb,chunk,4)
        allrows=[r for p in root.glob('*_c*.json') for r in json.loads(p.read_text())]
        assert len(allrows)==288
        result=dict(n_layer_step_events=288,heads=12,layers=LAYERS,split='val',
            nmse={m:sum(v for r in allrows for v in r['nmse'][m])/(288*12) for m in allrows[0]['nmse']},
            replaced_fraction=sum(v for r in allrows for v in r['replaced_fraction'])/(288*12),
            route_norm_ratio=sum(v for r in allrows for v in r['route_norm_ratio'])/(288*12))
        atomic(root/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    finally:engine.callback=None;lock.close()
if __name__=='__main__':run()
