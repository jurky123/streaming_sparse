"""Time clean page stores in the deployment layout and verify ring rollover."""
import json
import statistics
import torch
from ..review_v1.ops import Config,Summary
from ..review_v1.runtime import PageStore
from .profile_route import capture
from .weights import ROOT

@torch.no_grad()
def main():
    data=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
    k,v=(data[n][0,:4608].cuda().transpose(0,1).reshape(12,72,64,128).contiguous() for n in ('k','v'))
    summary=Summary(*(data['summary'][n][:,:72].cuda().contiguous() for n in ('keys','mean','detail','stats')))
    order=torch.arange(4608,device='cuda').expand(12,-1)
    rows=[]
    for major in (False,True):
        stores=[];graphs=[]
        for direct in (False,True):
            cache={n:torch.zeros(1,27648,12,128,device='cuda',dtype=k.dtype) for n in ('k','v')}
            if major:cache={n:x.view(1,12,27648,128).transpose(1,2) for n,x in cache.items()}
            store=PageStore(cache,Config(history_pages=360))
            store.contiguous_commit=direct
            def run():
                store.write=144
                store.commit(k,v,summary,order,4608)
            graph,_=capture(run)
            graphs.append(graph);stores.append(store)
        times=[[],[]]
        for rep in range(26):
            for i in (rep%2,1-rep%2):
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(12):graphs[i].replay()
                b.record();b.synchronize()
                if rep>=2:times[i].append(a.elapsed_time(b)/12)
        # Reset metadata and compare after a complete ring and one overwrite.
        for store in stores:
            store.reset()
            for pos in range(6):store.commit(k,v,summary,order,pos*4608)
        x,y=stores
        checks=[torch.equal(x.cache[n],y.cache[n]) for n in ('k','v')]
        checks.extend(torch.equal(getattr(x.summary,n),getattr(y.summary,n)) for n in ('keys','mean','detail','stats'))
        checks.extend([torch.equal(x.token_ids,y.token_ids),(x.write,x.n,x.commits)==(y.write,y.n,y.commits)])
        ms=list(map(statistics.median,times));row=dict(major=major,ms=ms,speedup=ms[0]/ms[1],checks=checks)
        print(row,flush=True);rows.append(dict(**row,samples_ms=times))
    path=ROOT/'artifacts/video_runtime_v1/contiguous_commit_v1';path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
