"""Actual pooling, risk and Reader weight interventions in the production runtime."""
import copy
import json
import sys
from pathlib import Path

import torch

folder=Path(__file__).resolve().parent
repo=folder.parents[1]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from clean_policy import attach
from scope import apply_selected
from local_experiments.broad_common import Engine
from local_experiments.review_v1.ops import RouteOps,haar
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime

outdir=folder/'module_sensitivity'
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
    cfg=config('intersection_fast',local_assets=repo)
    configure(cfg)
    engine=Engine()
    for block in engine.wrapper.model.blocks:block.self_attn.sink_size=1
    model,reader,phi,B,_=load_weights(cfg)
    layers=active_layers if case.endswith('_scope') else tuple(range(1,30))
    heads=active_heads if case.endswith('_scope') else tuple(range(12))
    clean_models={}
    if case.startswith('summary'):
        state=torch.load(prior/'summary.pt',map_location='cuda',weights_only=False)['model']
        for l in layers:
            layer=model[l]
            if case=='summary_clean':
                layer=copy.deepcopy(layer)
                clean_models[id(model[l])]=layer
            for field in ('pool_k','pool_v','pool_feature'):
                param=getattr(layer,field)
                param[list(heads)]=state[f'layers.{l}.{field}'][list(heads)]
    reader_cell={'reader_scope':(5,3),'reader_local_best':(17,3),'reader_local_worst':(27,9)}.get(case)
    original_reader=copy.deepcopy(reader) if reader_cell else None
    if case.startswith('reader'):
        reader.load_state_dict(torch.load(prior/'reader.pt',map_location='cuda',weights_only=False)['model'])
    rt=make_runtime(engine,model,reader,phi,B,cfg)
    rt.sparse_clean=mode=='sparse'
    gen=attach(rt,engine,10)
    apply_selected(rt,folder)
    if case=='summary_clean':
        build=rt.summary_graph
        def clean_build(layer,*args,**kwargs):
            if engine.current.get('step')==4:
                layer=clean_models.get(id(layer),layer)
            return build(layer,*args,**kwargs)
        rt.summary_graph=clean_build
    if original_reader is not None:
        # Use the tuned shared network only for the chosen cell at the actual
        # reader readout. Both encoders see exactly the same page features.
        encode=rt.encode
        def scoped_encode(l,s):
            rt.reader=original_reader
            base=encode(l,s)
            if l==reader_cell[0]:
                rt.reader=reader
                alt=encode(l,s)
                base['co'][reader_cell[1]].copy_(alt['co'][reader_cell[1]])
            return base
        rt.encode=scoped_encode
    stages=5 if rt.sparse_clean else 4
    mult=torch.ones(9,4,device='cuda')
    for j in range(4):
        mult[1+2*j,j]=.5
        mult[2+2*j,j]=2.
    weights=torch.stack([model[max(l,1)].theta for l in range(30)])
    if case=='fit':
        metric=torch.zeros(5,30,12,9,device='cuda',dtype=torch.float64)
        def sample(l,step,q,k,v,actual):
            c=engine.current['chunk']
            if c not in (2,4,6):return actual
            pages=k.size(1)//64
            h=torch.arange(12,device='cuda')
            row=(h*113+c*67+step*43+l*79)%4608
            query=q[0,row,h].float()
            s=rt.slabs[l].slice(0,pages)
            f,_=RouteOps.features(query[:,None],s,block=pages)
            scores=torch.einsum('hpf,hf,jf->hpj',f[:,0],weights[l],mult)
            keys=k[0].view(pages,64,12,128).permute(2,0,1,3)
            vals=v[0].view(pages,64,12,128).permute(2,0,1,3)
            dots=torch.einsum('hd,hptd->hpt',query,keys.float())/128**.5
            mass=dots.logsumexp(-1).softmax(-1)
            exact=torch.einsum('hpt,hptd->hpd',dots.softmax(-1),vals.float())
            z=torch.einsum('hd,hprd->hpr',query,s.keys.float())/128**.5
            summary_mass=z.logsumexp(-1).softmax(-1)
            codes=torch.einsum('hpr,ar,hpak->hpk',z.softmax(-1),haar(query),s.detail.float())
            approx=s.mean.float()+torch.einsum('hpk,hdk->hpd',codes,model[l].basis.bfloat16().float())
            center=(mass[...,None]*exact).sum(1,keepdim=True)
            residual=(mass[...,None]*(exact-center)-summary_mass[...,None]*(approx-center)).norm(dim=-1)
            for sl in (slice(24,pages-72),slice(pages-72,pages)):
                x=scores[:,sl];y=residual[:,sl]
                x=x-x.mean(1,keepdim=True);y=y-y.mean(1,keepdim=True)
                metric[step,l]+=torch.nn.functional.cosine_similarity(x,y[...,None],dim=1)
            return actual
        rt.sparse_output_probe=sample
        for p in ('forest_run','kitchen_cook'):
            for seed in (113,557):
                rt.reset();gen(embedding(p),seed,7)
        torch.save((weights[None]*mult[metric.argmax(-1)]).cpu(),outdir/f'risk_{mode}.pt')
        return
    if case.startswith('risk'):
        fitted=torch.load(outdir/f'risk_{mode}.pt',weights_only=True).cuda()
        original=rt.forward
        new=weights.unsqueeze(0).repeat(5,1,1,1)
        steps=([4] if case=='risk_clean' else [0] if case.endswith('_scope') else list(range(stages)))
        for st in steps:
            for l in layers:new[st,l,list(heads)]=fitted[st,l,list(heads)]
        logtheta=new.expm1().log()
        def forward(l,*args):
            model[l].logtheta.copy_(logtheta[engine.current['step'],l])
            return original(l,*args)
        rt.forward=forward
    # Force dense outputs after measuring each actual sparse operator. Subsequent
    # layers and chunks thus receive matched inputs across all interventions.
    sums=torch.zeros(5,30,12,2,device='cuda',dtype=torch.float64)
    def measure(l,st,q,k,v,actual):
        target=engine.dense(q,k,v)
        rows=slice((l*13+st*7+engine.current['chunk']*11)%64,None,64)
        ref=target[:,rows].float()
        sums[st,l,:,0]+=((actual[:,rows].float()-ref)**2).sum((0,1,3))
        sums[st,l,:,1]+=(ref**2).sum((0,1,3))
        return target
    local=None
    if not long:
        rt.sparse_output_probe=measure
        for p,seed in tests:
            rt.reset();gen(embedding(p),seed,7)
        local=sums.cpu().numpy()
        import numpy as np
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
        z=gen(embedding(p),seed,41 if long else 7)
        mse=float((z[:,9:].float()-ref[:,9:].float()).square().mean())
        def pixels(x):return np.concatenate(decode(vae,x,480 if long else 80))
        a=pixels(z)
        cache=outdir/f'ref_{p}_{seed}{label}.npy'
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
        rows.append(dict(prompt=p,seed=seed,mse=mse,lpips=float(np.mean(segments))))
        (outdir/f'{mode}_{case}{label}.json').write_text(json.dumps(rows))
        print(case,rows[-1],flush=True)

run(*sys.argv[1:3],long='--long' in sys.argv)
