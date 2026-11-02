"""Post-training diagnostics and microtiming; does not select or train models."""
import json
from pathlib import Path
import torch
from .learned_cache.data import Bank, records
from .learned_cache.model import Prototype, Corrector, predict, initial_mask, alternatives, output_for_masks
from .learned_cache.run import nmse
from .learned_cache.io import atomic
from .cache_grouping import history_order

ROOT=Path('artifacts/learned_cache_pilot_20261029')

@torch.no_grad()
def main():
    model=Prototype().cuda();corrector=Corrector().cuda()
    model.load_state_dict(torch.load(ROOT/'prototype/best.pt',weights_only=False,map_location='cuda')['model'])
    corrector.load_state_dict(torch.load(ROOT/'corrector/best.pt',weights_only=False,map_location='cuda')['corrector'])
    bank=Bank();rows=[]
    for path in records('test'):
        e=bank.get(path);s=e['hist']['state']
        proto=(s['k'].mean(2)[:,:,None],s['v'].mean(2)[:,:,None],torch.full((*s['k'].shape[:2],1),64.,device='cuda'))
        z,_,_=predict(e['route'],proto,e['kc'],e['vc'])
        out=output_for_masks(e['z'],e['values'],initial_mask(z)[:,:,None])[:,:,0]
        for i,head in enumerate(e['hist']['heads']):
            rows.append(dict(scene=e['case'],chunk=e['chunk'],layer=e['hist']['layer'],head=int(head),step=e['step'],
                             method='mean_k',nmse=nmse(out,e['target'])[i].mean().item()))
    atomic(ROOT/'mean_k.json',rows)
    e=bank.get(records('test')[0]);h=e['hist'];s=h['state']
    def timed(fn):
        for _ in range(5):fn()
        torch.cuda.synchronize();samples=[]
        for _ in range(30):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
        return dict(median_ms=sorted(samples)[15],min_ms=min(samples),max_ms=max(samples))
    timing={}
    timing['group_history_5chunks']=timed(lambda:history_order(h['k'],h['v'],0.))
    for learned in (False,True):
        prefix='learned' if learned else 'fixed'
        timing[prefix+'_summarize_5chunks']=timed(lambda:model(s,h['layer'],h['heads'],learned))
        protos=model(s,h['layer'],h['heads'],learned)
        def route(action):
            z,v,_=predict(e['route'],protos,e['kc'],e['vc']);mask=initial_mask(z)
            if action!='none':
                masks,f,g=alternatives(z,v,mask)
                score=corrector(f,g) if action=='learned' else g
                choice=score.argmax(-1)
                mask=masks.gather(2,choice[:,:,None,None].expand(-1,-1,1,masks.shape[-1]))
            return mask
        for action in ['none','analytic']+(['learned'] if learned else []):
            timing[prefix+'_'+action+'_routing']=timed(lambda:route(action))
    timing['scope']=dict(heads=3,queries=64,historical_tokens=23040,current_tokens=4608,
                        includes_sparse_attention_kernel=False,includes_q_tile_mean=False,
                        note='CUDA microtimings on one event; no video throughput claim')
    timing['auxiliary_bytes_per_layer_3heads']=dict(
        fp32_prototype_kv=sum(x.numel()*x.element_size() for x in protos[:2]),
        fp32_counts=protos[2].numel()*protos[2].element_size(),
        permutation_int32=3*23040*4,membership_fp32=3*23040*4*4,
        prototype_parameters=sum(p.numel()*p.element_size() for p in model.parameters()),
        corrector_parameters=sum(p.numel()*p.element_size() for p in corrector.parameters()))
    atomic(ROOT/'timing.json',timing)
    rows=[]
    for p in (ROOT/'evaluation').glob('*.json'):
        value=json.loads(p.read_text())
        if isinstance(value,list):rows.extend(value)
    rows.extend(json.loads((ROOT/'mean_k.json').read_text()))
    groups={}
    for field in ['scene','layer','head']:
        groups[field]={}
        for value in sorted({r[field] for r in rows}):
            groups[field][str(value)]={m:sum(r['nmse'] for r in rows if r[field]==value and r['method']==m)/
                sum(1 for r in rows if r[field]==value and r['method']==m) for m in sorted({r['method'] for r in rows})}
    atomic(ROOT/'breakdown.json',groups)
    print(json.dumps(dict(timing=timing,breakdown=groups),indent=2))

if __name__=='__main__':main()
