import json,time,statistics
from pathlib import Path
import torch
from scan import counts,GRID,ROOT
from phase import set_phase
from anchor_policy import install
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1 import legacy_components
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
from local_experiments.video_runtime_v1.accepted_tiers import config

def ms(fn):
    fn();torch.cuda.synchronize()
    samples=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(6): fn()
        b.record(); b.synchronize(); samples.append(a.elapsed_time(b)/6)
    return statistics.median(samples)

@torch.no_grad()
def run():
    start=time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=config('intersection',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine();engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    install(rt,'frame_fixed15')
    orig=legacy_components.attention
    replay=rt.batch_graph.replay; risk=None
    def observe(*args):
        nonlocal risk
        ret=replay(*args);risk=ret[2];return ret
    rt.batch_graph.replay=observe
    rows=[]
    def bench(q,k,v,s,basis,ids,workspace):
        p=s.mean.shape[1]
        c=engine.current['chunk']
        if engine.current['layer']!=1 or engine.current['step']!=0 or c==5:
            return engine.dense(q,k,v)
        ref=engine.dense(q,k,v)
        refms=ms(lambda:engine.dense(q,k,v))
        df=4*12*4608*p*64*128
        own=torch.arange(72,device=q.device)[None,:,None].expand(12,-1,-1)
        phase_start=time.monotonic()
        def split(a,b):
            hist=risk[...,24:p-72].topk(b,dim=-1).indices+24
            cur=risk[...,p-72:].scatter(-1,own,torch.inf).topk(a,dim=-1).indices+p-72
            anchor=torch.arange(24,device=q.device).expand(12,72,-1)
            return torch.cat((cur[...,:1],anchor,cur[...,1:],hist),-1).int().contiguous()
        for spec,(a,b) in zip(GRID,counts(p)):
            budget=24+a+b
            selected=split(a,b)
            assert (selected.sort(-1).values.diff()>0).all()
            f=lambda:orig(q,k,v,s,basis,selected,workspace)
            result=f().clone()
            tm=ms(f);sel=ms(lambda:split(a,b))
            padded=(p+15)//16*16
            ef=6*12*4608*64*128*budget
            tf=2*12*4608*padded*(4*128+2*128+2*3*16)
            predicted=(ef+tf)/df*refms
            rows.append(dict(pages=p,spec=spec,current=a,history=b,exact=budget,dense_ms=refms,select_ms=sel,fused_ms=tm,
                             nmse=float((result.float()-ref.float()).square().sum()/ref.float().square().sum()),
                             dense_flops=df,exact_flops=ef,comp_flops=tf,at_dense_rate_ms=predicted,gap_ms=tm-predicted))
        print(p,round(time.monotonic()-phase_start,2),flush=True)
        (ROOT/'module_rows.json').write_text(json.dumps(rows,indent=2))
        return ref
    legacy_components.attention=bench
    rt.reset()
    emb=torch.load(Path('artifacts/e24_joint_representatives_v1/embeddings/forest_run.pt'),weights_only=True).cuda()
    for c in range(7):
        if c==6:
            canonical=torch.cat([x['k'][:,:1536] for x in engine.cache],0);set_phase(rt,canonical,3)
        generate_chunk(engine,emb,113,c)
    (ROOT/'module_wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,rows=len(rows))))
run()
