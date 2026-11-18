"""Paired city_walk timing, verifying that the actual projected K/V suffix repeats."""
import json
from pathlib import Path
import time
import argparse
import torch
from rounding import rounded
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime
from local_experiments.video_runtime_v1.weights import ROOT

MODE=121

def fast(q,k,v):
    b,n,h,d=q.shape
    out=torch.empty_like(q)
    rounded[(n//64,h)](q,k,v,out,h,d,64,d**-.5*1.4426950408889634,MODE,num_warps=4,enable_fp_fusion=False)
    return out

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plain",action="store_true")
    options=parser.parse_args()
    from measure import compressed
    import wan.modules.model as model
    torch.set_num_threads(4)
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    configure(cfg)
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    original=model.flash_attention
    seen=set()
    def checked(q,k,v,*args,**kw):
        if k.size(1)!=512:return original(q,k,v,*args,**kw)
        if verify and engine.current['layer'] not in seen:
            assert all(torch.equal(x[:,63:],x[:,63:64].expand_as(x[:,63:])) for x in (k,v))
            seen.add(engine.current['layer'])
        if options.plain:
            out=torch.empty_like(q)
            compressed[(72,12)](q,k,v,out,12,128,64,128**-.5*1.4426950408889634,num_warps=4)
            return out
        return fast(q,k,v)
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    rows=[]
    for pair in range(-1,4):
        times=[0.,0.]
        outputs=[None,None]
        verify=pair<0
        for i in ([0,1] if pair%2==0 else [1,0]):
            model.flash_attention=checked if i else original
            rt.reset()
            torch.cuda.synchronize()
            start=time.perf_counter()
            latent=torch.cat([generate_chunk(engine,emb,701,c) for c in range(9)],dim=1)
            torch.cuda.synchronize()
            times[i]=time.perf_counter()-start
            outputs[i]=latent.cpu()
        assert len(seen)==30
        if pair>=0:
            old,new=outputs
            delta=new.float()-old.float()
            row=dict(old_s=times[0],new_s=times[1],max_abs=float(delta.abs().max()),nmse=float(delta.square().sum()/old.float().square().sum()),equal=torch.equal(old,new))
            rows.append(row)
            Path(__file__).with_name('base_plain.json' if options.plain else 'base.json').write_text(json.dumps(rows,indent=2))
            if pair==0:torch.save(outputs,'/tmp/jzk_cross_plain_latents.pt' if options.plain else '/tmp/jzk_cross_latents.pt')
            print(row,flush=True)
    rt.close()
if __name__=='__main__':main()
