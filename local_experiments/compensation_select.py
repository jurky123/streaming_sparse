"""Validation selection only; strength shrinkage is part of the frozen method."""
import json
from pathlib import Path
import numpy as np
from .learned_cache.io import atomic,digest
from .compensation_probe import ROOT

def main():
    assert json.loads((ROOT/'status.json').read_text())['stage']=='complete'
    rows=[r for p in sorted((ROOT/'val').glob('*.json')) for r in json.loads(p.read_text())['rows']]
    candidates={.25:'raw_rank8V_g0.25',.5:'raw_rank8V_g0.5',1.:'raw_rank8V'}
    gamma={};details=[];chosen_errors=[];base_errors=[]
    for l in range(1,30):
        layer=[r for r in rows if r['layer']==l];gamma[str(l)]=[]
        for h in range(12):
            base=np.array([r['methods']['hard_drop']['query_nmse'][h] for r in layer]);best=base;strength=0.
            for g,name in candidates.items():
                cur=np.array([r['methods'][name]['query_nmse'][h] for r in layer])
                # Mean gain plus bounded tail and each-prompt degradation on validation.
                safe=all(cur[[r['case']==c for r in layer]].mean()<=base[[r['case']==c for r in layer]].mean()*1.10+1e-8 for c in {r['case'] for r in layer})
                safe &= np.quantile(cur,.99)<=np.quantile(base,.99)*1.10+1e-8
                if safe and cur.mean()<base.mean()*.98 and cur.mean()<best.mean():best=cur;strength=g
            gamma[str(l)].append(strength);chosen_errors.extend(best.flatten().tolist());base_errors.extend(base.flatten().tolist())
            details.append(dict(layer=l,head=h,gamma=strength,base=float(base.mean()),selected=float(best.mean())))
    result=dict(gamma=gamma,details=details,enabled_heads=sum(x>0 for gs in gamma.values() for x in gs),validation_nmse=float(np.mean(chosen_errors)),validation_baseline_nmse=float(np.mean(base_errors)),weights_sha256=digest(ROOT/'model.pt'),source_sha256=digest(__file__),policy='Select from gamma0/.25/.5/1 on four validation prompts; >=2% per-head mean gain, <=10% per-prompt mean and p99 worsening; validate on fresh-seed full videos. This is fitted validation performance, not heldout improvement.',runtime_variant='static-centered BF16 rank8 V codes; raw positive 4-K-prototype partition estimate; no learned mass affine; fixed original top72 mask')
    atomic(ROOT/'selection.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['gamma','details']}),flush=True)
if __name__=='__main__':main()
