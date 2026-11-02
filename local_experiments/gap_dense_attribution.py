"""Measure the attention work eligible for replacement across one whole video."""
import torch,json,time
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk,timed_run
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from utils.wan_wrapper import WanVAEWrapper
    import wan.modules.causal_model as causal
    engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda();rt=Runtime(engine,'dense')
    original=causal.attention;events=[]
    try:
        for c in range(6):z=generate_chunk(engine,emb,101,c)
        vae.model.clear_cache();vae.decode_to_pixel(z,use_cache=True);vae.decode_to_pixel(z,use_cache=True)
        def observed(q,k,v,*args,**kwargs):
            meta=engine.current.copy();is_self=q.shape[1]==4608 and k.shape[1]>=4608
            eligible=is_self and meta['layer']>0 and meta['chunk']>=2 and meta['step']<4
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            out=original(q,k,v,*args,**kwargs);b.record();events.append((eligible,is_self,meta,a,b))
            return out
        causal.attention=observed
        zs,frames,perf=timed_run(engine,rt,vae,emb,101);assert not device_peers()
        rows={}
        for eligible,is_self,meta,a,b in events:
            key='eligible' if eligible else 'other_self' if is_self else 'other'
            row=rows.setdefault(key,dict(calls=0,total_ms=0.));row['calls']+=1;row['total_ms']+=a.elapsed_time(b)
        assert rows['eligible']['calls']==39*4*29,rows
        removable=rows['eligible']['total_ms']/1000;total=perf['total_generation_vae_seconds']
        atomic('artifacts/gap_pool_20261031/dense_attribution.json',dict(stages=rows,performance=perf,zero_cost_replacement_estimate_seconds=total-removable,zero_cost_speedup_estimate=total/(total-removable),scope='CUDA-event timing of every actual dense attention eligible for sparse replacement. Counterfactual zero routing/summary/exact/tail cost; optimistic Amdahl estimate, not achievable theoretical target. Event/host overhead perturbs wall time; no hardware-counter proof.'))
        print(json.dumps(dict(stages=rows,total=total,zero_cost_estimate=total-removable)),flush=True)
    finally:causal.attention=original;rt.close()
if __name__=='__main__':main()
