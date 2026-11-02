"""Supervised finite policy fitting; no weight gradient or validation selection."""
import json
from pathlib import Path
from .high_sparse_output_runtime import POLICIES
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_twostep_20261031')
def main():
    data={split:[json.loads(p.read_text()) for p in sorted((ROOT/split).glob('*.json'))] for split in ['train','val']}
    assert len(data['train'])==8 and len(data['val'])==4
    means={};chosen={}
    for step in [0,2]:
        rows=[r['rows'][0] for r in data['train'] if r['rows'][0]['step']==step];assert len(rows)==4
        means[str(step)]={p:sum(r['metrics'][p]['two']['nmse'] for r in rows)/len(rows) for p in POLICIES}
        chosen[str(step)]=min(means[str(step)],key=means[str(step)].get)
    evaluation={}
    for split,samples in data.items():
        sums={'off':0.,'schedule':0.}
        for s in samples:
            r=s['rows'][0];m=r['metrics'];sums['off']+=m['off']['two']['nmse'];sums['schedule']+=m[chosen[str(r['step'])]]['two']['nmse']
        evaluation[split]={k:v/len(samples) for k,v in sums.items()}
    atomic(ROOT/'schedule_selection.json',dict(policies=[chosen['0'],chosen['2']],training_means=means,evaluation=evaluation,scope='Finite hard gate policy fitting on train two-step x0 loss; 4 training cases per pair; validation not used for selection; no gradient training'))
    print(json.dumps(dict(policies=[chosen['0'],chosen['2']],evaluation=evaluation)))
if __name__=='__main__':main()
