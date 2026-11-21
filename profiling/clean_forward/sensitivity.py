"""Fit per-head output-projection gains by least squares and test placement in closed loop."""
import json
import sys
from pathlib import Path

import torch

folder = Path(__file__).resolve().parent
repo = folder.parents[1]
sys.path.insert(0, str(repo / 'reproduction/vbench_current'))
from phase import set_phase
from clean_policy import attach
from scope import apply_selected
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure, make_runtime, load_weights
from local_experiments.video_runtime_v1.backbone_ops import install

def embed(prompt):
    path = folder/'embeddings'/f'{prompt}.pt'
    if prompt=='short_721':
        path=repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt'
    return torch.load(path,weights_only=True).cuda()

@torch.no_grad()
def run(mode):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 512
    cfg = config('intersection_fast',local_assets=repo)
    configure(cfg)
    engine = Engine()
    for block in engine.wrapper.model.blocks:
        block.self_attn.sink_size = 1
    tests = [(p,s) for p in ('snow_ski','bird_flight') for s in (917,919)]
    confirm = '--confirm' in sys.argv
    if confirm:
        tests=[(p,s) for p in ('city_walk','short_721') for s in (721,722)]
    long = '--long' in sys.argv
    chunks = 41 if long else 7
    suffix='_30s' if long else ''
    def reference_path(p,s):
        return folder/f'sensitivity_ref_{p}_{s}{suffix}.pt'
    if mode=='reference':
        from types import SimpleNamespace
        install(engine,cfg)
        rt = SimpleNamespace(engine=engine,stores={})
        for p,s in tests:
            engine.reset()
            z=[]
            emb=embed(p)
            for c in range(chunks):
                if c==6:
                    canonical=torch.cat([cache['k'][:,:1536] for cache in engine.cache])
                if c>=6:
                    set_phase(rt,canonical,3*c-15)
                z.append(generate_chunk(engine,emb,s,c))
            target=reference_path(p,s)
            tmp=target.with_suffix('.partial')
            torch.save(torch.cat(z,1).cpu(),tmp)
            tmp.replace(target)
        return
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    gen=attach(rt,engine,10)
    apply_selected(rt,folder)
    rt.sparse_clean=mode=='sparse'
    stats=torch.zeros(5,30,12,4,device='cuda',dtype=torch.float64)
    def collect(l,s,q,k,v,out):
        # Rotate sample locations across chunks and layers; score on identical Q/K/V.
        offset=(engine.current['chunk']*11+l*7+s*13)%64
        rows=slice(offset,None,64)
        x=out[:,rows].float()
        y=engine.dense(q[:,rows].contiguous(),k,v).float()
        stats[s,l]+=torch.stack([(x*x).sum((0,1,3)),(x*y).sum((0,1,3)),
                                 (y*y).sum((0,1,3)),((x-y)**2).sum((0,1,3))],-1)
        return out
    rt.sparse_output_probe=collect
    def rollout(p,s):
        rt.reset()
        return gen(embed(p),s,chunks)
    if long:
        gain=torch.load(folder/f'sensitivity_{mode}_gains.pt',weights_only=True).cuda()
        rt.sparse_output_probe=None
    else:
        for p in ('forest_run','kitchen_cook'):
            for s in (113,557):
                rollout(p,s)
        a,b,y,e=stats.unbind(-1)
        gain=torch.where(a>0,b/a,1.).float()
        def improvements(a,b,y,e):
            d=gain.double()-1
            return -(2*d*(a-b)+d*d*a)
        if '--clean-axis' not in sys.argv:
            torch.save(gain.cpu(),folder/f'sensitivity_{mode}_gains.pt')
        stats.zero_()
        for p,s in tests:
            rollout(p,s)
        delta=improvements(*stats.unbind(-1)).cpu()
        err=stats[...,3].cpu()
    report={}
    path=folder/f'sensitivity_{mode}.json'
    cases=[('off',[])]
    active=torch.zeros_like(gain, dtype=torch.bool).cpu()
    active[:4+(mode=='sparse'),1:]=True
    # Predefined single-axis interventions, shared by the two clean modes.
    for axis,n in [('step',5 if mode=='sparse' else 4),('layer',30),('head',12)]:
        for i in range(n):
            if axis=='layer' and i not in (1,5,14,15,27,29):continue
            if axis=='head' and i not in (0,3,4,6,9,11):continue
            mask=torch.zeros_like(active)
            if axis=='step':mask[i]=True
            if axis=='layer':mask[:,i]=True
            if axis=='head':mask[:,:,i]=True
            mask &=active
            cases.append((f'{axis}{i}',mask))
    # Interleave axes so partial results cover all three.
    cases=[cases[0]]+sorted(cases[1:],key=lambda row:(int(''.join(filter(str.isdigit,row[0]))),row[0]))
    refs={} if long else {case:torch.load(reference_path(*case),weights_only=True).cuda().float() for case in tests}
    if confirm:
        import numpy as np
        selection=json.loads(path.read_text())
        baseline=np.array(selection['off']['mse'])
        combined=active.clone()
        chosen={}
        for axis,dim,count in [('step',0,2),('layer',1,3),('head',2,3)]:
            rows=sorted((np.mean(np.array(v['mse'])/baseline),int(k[len(axis):]))
                        for k,v in selection.items() if k.startswith(axis))
            ids=[i for score,i in rows[:count] if score<1.]
            chosen[axis]=ids
            condition=torch.zeros(active.shape[dim],dtype=torch.bool)
            condition[ids]=True
            shape=[1,1,1];shape[dim]=-1
            combined &=condition.reshape(shape)
        print('selected',chosen,flush=True)
        import lpips
        from local_experiments.stream_eval_common import native_vae,decode
        from local_experiments.high_sparse_experiment import prefix_quality
        vae=native_vae()
        metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
        def rgb(z):
            return np.concatenate(decode(vae,z.bfloat16(),480 if long else 80))
        def perceptual(candidate,ref):
            scores=[]
            for i in range(0,len(ref),80):
                def segment(x):return [np.concatenate([x[:21],x[i:i+80]])]
                scores.append(prefix_quality(segment(candidate),segment(ref),metric)['lpips'])
            return float(np.mean(scores))
        reference_rgb={case:rgb(ref) for case,ref in refs.items()}
        cases=[('off',[]),('intersection',combined)]
        if mode=='sparse':
            clean=active.clone()
            clean[:4]=False
            cases.append(('clean_only',clean))
            restricted=torch.zeros_like(active)
            for layer in chosen['layer']:
                restricted[4,layer,chosen['head']]=True
            cases.append(('clean_layer_head',restricted))
        path=folder/f'sensitivity_{mode}_confirm{suffix}.json'
    if '--clean-axis' in sys.argv:
        assert mode=='sparse' and not confirm
        cases=[('off',[])]
        for axis,ids in [('layer',(1,27,29)),('head',(0,3,6))]:
            for i in ids:
                mask=torch.zeros_like(active)
                if axis=='layer':mask[4,i]=True
                if axis=='head':mask[4,1:,i]=True
                cases.append((f'clean_{axis}{i}',mask))
        path=folder/'sensitivity_clean_axis.json'
    if long:cases=[case for case in cases if case[0] in ('off','clean_only')]
    if path.exists():report=json.loads(path.read_text())
    for name,mask in cases:
        if name in report:continue
        weights=None if name=='off' else torch.where(mask.cuda(),gain,1.)
        def apply(l,s,q,k,v,out):
            return (out.float()*weights[s,l,:,None]).to(out.dtype)
        rt.sparse_output_probe=None if weights is None else apply
        scores=[]
        quality=[]
        for p,s in tests:
            case=(p,s)
            z=rollout(p,s).float()
            if case not in refs:
                import time
                target=reference_path(p,s)
                while not target.exists():time.sleep(2)
                time.sleep(1)
                refs[case]=torch.load(target,weights_only=True).cuda().float()
                reference_rgb[case]=rgb(refs[case])
            ref=refs[case]
            scores.append(float((z[:,9:]-ref[:,9:]).square().mean()))
            if confirm:
                quality.append(perceptual(rgb(z),reference_rgb[case]))
        report[name]=dict(mse=scores,local_reduction=None if long or weights is None or not mask.any() else float(delta[mask].sum()/err[mask].sum()))
        if confirm:report[name]['lpips480' if long else 'lpips80']=quality
        if confirm and name=='intersection':report[name]['selected']=chosen
        path.write_text(json.dumps(report))
        print(mode,name,report[name],flush=True)

run(sys.argv[1])
