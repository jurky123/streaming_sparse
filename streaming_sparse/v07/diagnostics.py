"""Fixed-query exact one-swap probe; diagnostic only, never emits training labels."""
import json,math
import torch
from .common import HERE,fixed_samples,atomic_json,source_identity,query_slices,output
from .data import data
from .exact import teacher,page_costs,singleton,set_gain
from ..v05.primitives import select


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=data('val');plan=fixed_samples(ds);rows=[];directory=output(HERE/'phase1/fixed_swap_probe')
    for i,s in enumerate(plan):
        events=ds.load(s['snapshot'],'cuda');e=next(e for e in events if e['step_id']==s['step'])
        sl=query_slices(e,16)[s['tile']];idx=sl.start+s['query'];q=e['q'][idx:idx+1]
        t=teacher(q,e);g=t.geometry();costs=page_costs(len(e['k_hist']),64,q.device);budget=math.floor(.2*len(e['k_hist']))
        score=singleton(g).clamp_min(1e-12).log()-costs.log()[None];mask=select(score,costs,budget)
        target=t.dense();base=float((t.output(mask)-target).square().sum());best=base;best_pair=None;candidates=0
        # Enumerate affordable one-swaps, batching all insertions for each removal.
        kept=mask[0].nonzero().flatten();removed=(1-mask[0]).nonzero().flatten();used=float((mask*costs).sum())
        for remove in kept:
            valid=used-costs[remove]+costs[removed]<=budget
            inserts=removed[valid]
            if not len(inserts):continue
            trial=mask.expand(len(inserts),-1).clone();trial[:,remove]=0;trial[torch.arange(len(inserts)),inserts]=1
            # Stable exact output for many masks of one query, without rebuilding KV.
            logits=t.logz.expand(len(inserts),-1);mu=t.mu.expand(len(inserts),-1,-1)
            from .exact import MaskOutput
            out=MaskOutput.apply(trial,logits,mu,t.current_logz.expand(len(inserts)),t.current_output.expand(len(inserts),-1))
            errors=(out-target).square().sum(-1);j=int(errors.argmin());candidates+=len(inserts)
            if float(errors[j])<best:best=float(errors[j]);best_pair=[int(remove),int(inserts[j])]
        rows.append(dict(**s,base_sse=base,best_one_swap_sse=best,candidates=candidates,best_pair=best_pair,
                         role='offline headroom probe only; not an oracle label or deployed refinement'))
        atomic_json(directory/'progress.json',dict(completed=i+1,total=len(plan)))
    atomic_json(directory/'summary.json',dict(source_sha256=source_identity(),samples=rows,split='val',no_training_labels=True))


if __name__=='__main__':main()
