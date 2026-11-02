"""CPU audit of coverage, finite training updates and validation selection."""
import json
import math
import torch
from .mass_common import ROOT,CHUNKS,cases
from .learned_cache.io import atomic
from .mass_router import METHODS

def run():
    results=[]
    for first,last in [(0,30)]:
        folder=ROOT/f'train_{first:02d}_{last:02d}'
        status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
        saved=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
        for epoch in range(1,7):
            seen=set();maxgrad=0.
            for c in cases('train'):
                for chunk in CHUNKS:
                    rows=json.loads((folder/'train'/f'epoch{epoch:02d}'/f'{c["id"]}_c{chunk:02d}.json').read_text())
                    for r in rows:
                        key=(c['id'],chunk,r['step'],r['layer'],r['method']);assert key not in seen
                        seen.add(key);assert math.isfinite(r['loss']) and math.isfinite(r['grad'])
                        maxgrad=max(maxgrad,r['grad'])
            expected={(c['id'],ch,s,l,m) for c in cases('train') for ch in CHUNKS for s in range(4) for l in range(first,last) for m in METHODS}
            assert seen==expected
            results.append(dict(first=first,last=last,epoch=epoch,updates=len(seen),max_preclip_grad=maxgrad))
        for epoch in range(7):
            rows=[]
            for c in cases('val'):
                for chunk in CHUNKS:rows+=json.loads((folder/'val'/f'epoch{epoch:02d}'/f'{c["id"]}_c{chunk:02d}.json').read_text())
            for l in range(first,last):
                for m in METHODS:
                    vals=[v for r in rows if r['layer']==l and r['method']==m for v in r['nmse']]
                    assert len(vals)==576
                    assert math.isclose(math.fsum(vals)/len(vals),saved['curves'][epoch]['scores'][m][l],rel_tol=1e-12,abs_tol=1e-14)
        for l in range(first,last):
            assert status['updates'][str(l)]==576
            for m in METHODS:
                assert saved['metrics'][m][l]==min(c['scores'][m][l] for c in saved['curves'])
    atomic(ROOT/'training_ledger_checks.json',dict(passed=True,rows=results))
    print('Complete matched training coverage, finite losses/gradients, validation scores and checkpoint selection verified')
if __name__=='__main__':run()
