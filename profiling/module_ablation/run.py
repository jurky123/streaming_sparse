"""Actual pooling, risk and Reader weight interventions in the production runtime."""
import time
import json
import sys
from pathlib import Path

import torch

folder=Path(__file__).resolve().parents[1]/'clean_forward'
sys.path.insert(0,str(folder))
repo=folder.parents[1]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from policy import attach
from scope import apply_selected
from local_experiments.broad_common import Engine
from local_experiments.review_v1.ops import RouteOps,haar
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime

outdir=Path(__file__).resolve().parent
outdir.mkdir(exist_ok=True)
prior=repo/'reproduction/vbench_current/history/conditional_budget_v1/budget_10'
active_layers=(3,4,5,8,14,17,27)
active_heads=(3,4,9)
tests=[(p,s) for p in ('city_walk','short_721') for s in (721,722)]

def embedding(prompt):
    path=folder/'embeddings'/f'{prompt}.pt'
    if prompt=='short_721':path=repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt'
    return torch.load(path,weights_only=True).cuda()

@torch.no_grad()
def run(mode,case,long=False):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.cache_size_limit=512
    cfg=config('base' if case=='base' else 'intersection_fast',local_assets=repo)
    configure(cfg)
    engine=Engine()
    for block in engine.wrapper.model.blocks:block.self_attn.sink_size=1
    model,reader,phi,B,_=load_weights(cfg)
    rt=make_runtime(engine,model,reader,phi,B,cfg)
    rt.sparse_clean=True
    if case=='base':
        from phase import set_phase
        from local_experiments.rollout_suite import generate_chunk
        def gen(emb,seed,chunks):
            parts=[]
            for c in range(chunks):
                if c==6:canonical=torch.cat([cache['k'][:,:1536] for cache in engine.cache])
                if c>=6:set_phase(rt,canonical,3*c-15)
                parts.append(generate_chunk(engine,emb,seed,c))
            return torch.cat(parts,1)
    else:
        gen=attach(rt,engine,10,case)
        apply_selected(rt,folder)
    from interventions import install
    install(rt,'ungrouped' if case=='trace_natural' else case)
    if case.startswith('trace_'):
        from page_diagnostic import PageDiagnostic
        rt.packed_attention=PageDiagnostic(rt)
    # Force dense outputs after measuring each actual sparse operator. Subsequent
    # layers and chunks thus receive matched inputs across all interventions.
    sums=torch.zeros(5,30,12,2,device='cuda',dtype=torch.float64)
    windows=torch.zeros((7,2),dtype=torch.float64,device='cuda')
    def measure(l,st,q,k,v,actual):
        target=engine.dense(q,k,v)
        rows=slice((l*13+st*7+engine.current['chunk']*11)%64,None,64)
        ref=target[:,rows].float()
        error=((actual[:,rows].float()-ref)**2).sum((0,1,3))
        energy=(ref**2).sum((0,1,3))
        sums[st,l,:,0]+=error
        sums[st,l,:,1]+=energy
        if '--windows' in sys.argv:windows[engine.current['chunk']]+=torch.stack((error.sum(),energy.sum()))
        return target
    local=None
    if not long:
        rt.sparse_output_probe=measure
        for p,seed in tests:
            rt.reset();gen(embedding(p),seed,7)
        local=sums.cpu().numpy()
        import numpy as np
        if case.startswith('trace_'):
            np.save(outdir/f'{case}_mass.npy',rt.packed_attention.values.cpu().numpy())
            return
        if '--windows' in sys.argv:
            np.save(outdir/f'{case}_windows.npy',windows.cpu().numpy())
            return
        np.save(outdir/f'{mode}_{case}_local.npy',local)
    rt.sparse_output_probe=None
    import lpips
    import numpy as np
    from local_experiments.stream_eval_common import native_vae,decode
    import torch.nn.functional as F
    metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    vae=native_vae()
    rows=[]
    label='_30s' if long else ''
    for p,seed in tests:
        ref=torch.load(folder/f'sensitivity_ref_{p}_{seed}{label}.pt',weights_only=True).cuda()
        rt.reset()
        emb=embedding(p)
        torch.cuda.synchronize();start=time.perf_counter()
        z=gen(emb,seed,41 if long else 7)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        mse=float((z[:,9:].float()-ref[:,9:].float()).square().mean())
        def pixels(x):return np.concatenate(decode(vae,x,480 if long else 80))
        a=pixels(z)
        cache=folder/'module_sensitivity'/f'ref_{p}_{seed}{label}.npy'
        if not cache.exists():
            tmp=outdir/f'ref_{p}_{seed}{label}_{mode}.tmp.npy'
            np.save(tmp,pixels(ref))
            tmp.replace(cache)
        b=np.load(cache,mmap_mode='r')
        segments=[]
        for i in range(0,len(a),4):
            def batch(x):
                tensor=torch.from_numpy(x[i:i+4].copy()).cuda().permute(0,3,1,2).float()/255
                return F.interpolate(tensor,size=(256,384),mode='bilinear',align_corners=False)*2-1
            segments.extend(metric(batch(a),batch(b)).flatten().cpu().tolist())
        rows.append(dict(prompt=p,seed=seed,seconds=elapsed,mse=mse,lpips=float(np.mean(segments))))
        (outdir/f'{mode}_{case}{label}.json').write_text(json.dumps(rows))
        print(case,rows[-1],flush=True)

run(*sys.argv[1:3],long='--long' in sys.argv)
