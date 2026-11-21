"""Natural paging with the original first-roll survivor identity rule (7 chunks)."""
import json
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.fixed15_layout import pack_range

def install(rt):
    group=rt.batched_group
    commit=rt.commit_jobs
    pending={}
    counts=[]
    def remember(jobs):
        if jobs[0][4]==4608:
            batches=group([(m,k,v) for _,m,k,v,_,_ in jobs])
            for (layer,m,k,v,start,enabled),(_,_,order) in zip(jobs,batches):
                # At c5 commit the ring overwrites pages 24:96; chunk 1
                # originally occupied slots 72:144. Its last 48 pages survive.
                ids=order.reshape(12,4608)[:,1536:].sort(1).values
                def keep(t):
                    return t[0].transpose(0,1).gather(1,ids[...,None].expand(-1,-1,128)).reshape(12,48,64,128)
                pending[layer]=(keep(k),keep(v),ids+start)
                counts.append((ids>=1536).sum(1))
        commit(jobs)
    rt.commit_jobs=remember
    forward=rt.forward
    def matched(l,module,x,grid,freqs,cache,start):
        assert start<=6*4608
        if start==6*4608 and l in pending:
            kg,vg,ids=pending.pop(l)
            store=rt.stores[l]
            assert store.n==360 and store.write==96
            slices=slice(96,144)
            for field,t in [('k',kg),('v',vg)]:
                cache[field][0,96*64:144*64].copy_(t.flatten(1,2).transpose(0,1))
            store.token_ids[:,slices].copy_(ids.reshape(12,48,64))
            pk,pv=rt.packed_attention.cache(store)
            pack_range[(48,12,16)](cache['k'],cache['v'],pk,pv,96)
            rt.packed_attention.summary_versions.pop(store,None)
            s=rt.summary_graph(rt.model[l],kg,vg)
            for f in ('keys','mean','detail','stats'):
                getattr(store.summary,f)[:,slices].copy_(getattr(s,f))
            if l in rt.pset_layers:rt.write_encoding(l,rt.encode(l,s),start=96)
            if not pending:
                value=torch.cat(counts).float().mean().item()/3072
                counts.clear()
                with Path(__file__).with_name('matched_overlap.jsonl').open('a') as f:
                    print(json.dumps(dict(mean_survivor_fraction=value)),file=f)
        return forward(l,module,x,grid,freqs,cache,start)
    rt.forward=matched
