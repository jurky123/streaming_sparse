import json
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure
from local_experiments.video_runtime_v1.profile_route import capture
from local_experiments.video_runtime_v1.weights import ROOT
from local_experiments.broad_common import Engine,noise

@triton.jit
def compressed(Q,K,V,O,H:tl.constexpr,D:tl.constexpr,M:tl.constexpr,Scale:tl.constexpr):
    m=tl.program_id(0)*M+tl.arange(0,M)
    h=tl.program_id(1)
    n=tl.arange(0,64)
    d=tl.arange(0,D)
    q=tl.load(Q+(m[:,None]*H+h)*D+d[None,:])
    k=tl.load(K+(n[None,:]*H+h)*D+d[:,None])
    v=tl.load(V+(n[:,None]*H+h)*D+d[None,:])
    z=tl.dot(q,k)
    maximum=tl.max(z,1)
    pad=tl.sum(tl.where(n[None,:]==63,z,0.),1)
    previous=tl.exp2((pad-maximum)*Scale)
    w=tl.exp2(z*Scale-maximum[:,None]*Scale)
    pad_value=tl.load(V+(63*H+h)*D+d)
    numerator=(448.*previous)[:,None]*pad_value[None,:]
    out=tl.dot(w.to(tl.bfloat16),v,numerator)
    den=448.*previous+tl.sum(w,1)
    tl.store(O+(m[:,None]*H+h)*D+d[None,:],out/den[:,None])

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    configure(config('base',local_assets='/tmp/streaming_sparse-0/runtime'))
    engine=Engine()
    import wan.modules.model as model
    original=model.flash_attention
    examples=[]
    def grab(q,k,v,*args,**kw):
        if engine.current['layer'] in (0,12,29):
            examples.append((q.clone(),k.clone(),v.clone()))
        return original(q,k,v,*args,**kw)
    model.flash_attention=grab
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    engine.forward(noise(701,0,0),emb,0,0)
    model.flash_attention=original
    assert len(examples)==3
    torch.save([tuple(x.cpu() for x in triple) for triple in examples],'/tmp/jzk_cross_three.pt')
    result=[]
    for q,k,v in examples:
        assert all(torch.equal(x[:,63:],x[:,63:64].expand_as(x[:,63:])) for x in (k,v))
        def fast(m):
            out=torch.empty_like(q)
            compressed[(4608//m,12)](q,k,v,out,12,128,m,128**-.5*1.4426950408889634,
                                     num_warps=4,enable_fp_fusion=True)
            return out
        cases=[capture(lambda:original(q,k,v))]+[capture(lambda:fast(m)) for m in (16,32,64)]
        times=[[] for _ in cases]
        for rep in range(28):
            for j in range(4):
                i=(rep+j)%4
                a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                for _ in range(8):cases[i][0].replay()
                b.record();b.synchronize()
                if rep>=4:times[i].append(a.elapsed_time(b)/8)
        ref=cases[0][1].float()
        rows=[]
        for (_,out),sample in zip(cases,times):
            delta=out.float()-ref
            rows.append(dict(ms=statistics.median(sample),nmse=float(delta.square().sum()/ref.square().sum()),
                             max_abs=float(delta.abs().max()),changed=int(delta.count_nonzero()),samples=sample))
        result.append(rows)
        Path(__file__).with_name('module.json').write_text(json.dumps(result,indent=2))
        print([(r['ms'],r['nmse'],r['changed']) for r in rows],flush=True)
if __name__=='__main__':main()
