import sys,time,json,math
from pathlib import Path
import torch
import numpy as np
from anchor_policy import install
from phase import set_phase
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1 import legacy_components
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.video_runtime_v1.accepted_tiers import config

ROOT=Path(__file__).parent
GRID=[(t,s) for t in (5,8,10,12,15,20) for s in (0,.25,.5,.75,1)]+[('full',5),('full',10)]
def counts(p):
    hist=p-96
    result=[]
    for total,share in GRID:
        if total=='full':
            a,b=72,hist*share//100
        else:
            n=(p-24)*total//100
            a=min(72,max(1,n-hist,round(n*share)))
            b=n-a
        result.append((a,b))
    return result

@torch.no_grad()
def main(prompt="forest_run",seed=113):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('intersection',profile=True,local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    install(rt,'frame_fixed15')
    emb=torch.load(Path('artifacts/e24_joint_representatives_v1/embeddings')/(prompt+'.pt'),weights_only=True).cuda()
    rt.reset()
    data=torch.zeros(41,4,29,12,len(GRID),device='cuda')
    denom=torch.zeros_like(data[...,0])
    timer={}
    def timed(name,fn):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); out=fn(); b.record(); timer.setdefault(name,[]).append((a,b))
        return out
    risk=None
    replay=rt.batch_graph.replay
    def observe(*args):
        nonlocal risk
        ret=replay(*args); risk=ret[2]; return ret
    rt.batch_graph.replay=observe
    def scan(q,k,v,s,basis,ids,workspace,**kw):
        c,st,l=[engine.current[key] for key in ('chunk','step','layer')]
        p=s.mean.shape[1]; h=torch.arange(12,device=q.device)
        tile=(c*11+st*17+l*7+h*13)%72
        row=tile*64+(c*5+l*3+st*9+h*7)%64
        query=q[0,row,h].float()
        def stats():
            keys=k[0].reshape(p,64,12,128).permute(2,0,1,3)
            values=v[0].reshape(p,64,12,128).permute(2,0,1,3)
            dot=(keys.float()*query[:,None,None,:]).sum(-1)/math.sqrt(128)
            z=(s.keys.float()*query[:,None,None,:]).sum(-1)/math.sqrt(128)+math.log(16.)
            le=dot.logsumexp(-1); la=z.logsumexp(-1)
            ev=(dot.softmax(-1)[...,None]*values.float()).sum(2)
            w=z.softmax(-1)
            signs=query.new_tensor([[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])
            code=((w@signs.T)[...,None]*s.detail.float()).sum(2)
            av=s.mean.float()+code@basis.bfloat16().float().transpose(1,2)
            de=(le.softmax(-1)[...,None]*ev).sum(1)
            return (le,ev),(la,av),de
        e,a,dense=timed('page_statistics',stats)
        def evaluate():
            r=risk[h,tile]
            own=p-72+tile
            histories=r[:,24:p-72].argsort(stable=True,descending=True)+24
            current=r[:,p-72:].scatter(1,tile[:,None],float('inf')).argsort(stable=True,descending=True)+p-72
            cr=current.argsort(-1)
            hr=histories.argsort(-1)
            ac,hc=zip(*counts(p))
            ac=torch.tensor(ac,device=q.device)[:,None]
            hc=torch.tensor(hc,device=q.device)[:,None]
            mask=torch.cat((torch.ones(12,len(GRID),24,device=q.device,dtype=torch.bool),
                            hr[:,None]<hc,cr[:,None]<ac),2)
            le,ev=e;la,av=a
            prob=torch.where(mask,le[:,None],la[:,None]).softmax(-1)
            pred=(prob*mask)@ev+(prob*~mask)@av
            data[c,st,l-1]=((pred-dense[:,None])**2).sum(-1)
            denom[c,st,l-1]=dense.square().sum(-1)
        timed('all_budget_predictions',evaluate)
        return timed('dense_teacher_attention',lambda:engine.dense(q,k,v))
    legacy_components.attention=scan
    start=time.monotonic(); chunks=[]
    for c in range(41):
        t=time.monotonic()
        if c>=6:
            if c==6: canonical=torch.cat([x['k'][:,:1536] for x in engine.cache],0)
            timed('anchor_rebase',lambda:set_phase(rt,canonical,3*c-15))
        z=generate_chunk(engine,emb,seed,c)
        assert z.isfinite().all()
        torch.cuda.synchronize()
        chunks.append(time.monotonic()-t)
        print(c,round(chunks[-1],3),flush=True)
    elapsed=time.monotonic()-start
    scores=data[2:].sum((0,1,2,3))/denom[2:].sum()
    print("coverage",int((denom[2:]==0).sum()),"nonfinite",int((~data.isfinite()).sum()),flush=True)
    np.savez_compressed(ROOT/f'{prompt}_{seed}.npz',sse=data.cpu().numpy(),dense_energy=denom.cpu().numpy())
    assert (denom[2:]>0).all() and data.isfinite().all()
    results=dict(grid=GRID,counts={p:counts(p) for p in (216,288,360,432)},nmse=scores.tolist(),wall_seconds=elapsed,
                 chunk_seconds=chunks,device_ms={n:sum(a.elapsed_time(b) for a,b in pairs) for n,pairs in timer.items()},
                 stages=rt.timings.finish())
    (ROOT/f'{prompt}_{seed}.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__': main(sys.argv[1],int(sys.argv[2])) if len(sys.argv)>1 else main()
