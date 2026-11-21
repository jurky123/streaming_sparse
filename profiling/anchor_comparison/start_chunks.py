"""Paired startup-window test, original weights and dense clean."""
import json,sys,time,statistics
from pathlib import Path
import torch
folder=Path(__file__).resolve().parent
repo=folder.parents[1]
sys.path.insert(0,str(repo/'reproduction/vbench_current'))
from queue_policy import attach
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
@torch.no_grad()
def run():
    cfg=config('intersection_fast',local_assets=repo)
    torch.set_num_threads(3)
    configure(cfg)
    engine=Engine()
    for b in engine.wrapper.model.blocks:b.self_attn.sink_size=1
    rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    generate=attach(rt,engine,10)
    base=rt.forward
    events=[]
    def timed(l,*args):
        c,s=engine.current['chunk'],engine.current['step']
        if l==1 and s!=4:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();result=base(l,*args);b.record();events.append((c,a,b))
            return result
        return base(l,*args)
    rt.forward=timed
    data={}
    for p in ('city_walk','short_721'):
        embpath=repo/'profiling/clean_forward/embeddings/city_walk.pt' if p=='city_walk' else repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt'
        emb=torch.load(embpath,weights_only=True).cuda()
        for seed in (721,722):
            variants=[('dense',()),('early',(0,1)),('second',(1,))]
            for name,scope in variants if seed==721 else reversed(variants):
                rt.sparse_start_chunks=scope
                for rep in range(2):
                    rt.reset();events.clear();parts=[];seconds=[]
                    for c in range(2):
                        torch.cuda.synchronize();t=time.monotonic()
                        parts.append(generate_chunk(engine,emb,seed,c));torch.cuda.synchronize()
                        seconds.append(time.monotonic()-t)
                    assert torch.cat(parts,1).isfinite().all()
                row=dict(seconds=seconds,layer1_ms=[statistics.mean(a.elapsed_time(b) for i,a,b in events if i==c) for c in range(2)])
                ref=torch.load(repo/f'profiling/clean_forward/sensitivity_ref_{p}_{seed}.pt',weights_only=True).cuda()
                row['mse']=[(z.float()-ref[:,i*3:i*3+3].float()).square().mean().item() for i,z in enumerate(parts)]
                if '--video' in sys.argv:
                    rt.forward=base
                    rt.reset()
                    z=generate(emb,seed,7)
                    rt.reset();torch.cuda.synchronize();t=time.monotonic()
                    z=generate(emb,seed,7);torch.cuda.synchronize()
                    row['wall5']=time.monotonic()-t
                    row['future_mse']=float((z[:,6:].float()-ref[:,6:].float()).square().mean())
                    from local_experiments.stream_eval_common import native_vae,decode
                    import lpips,numpy as np
                    import torch.nn.functional as F
                    if 'vae' not in locals():
                        vae=native_vae();metric=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
                    pixels=np.concatenate(decode(vae,z,80))
                    reference=np.load(repo/f'profiling/clean_forward/module_sensitivity/ref_{p}_{seed}.npy',mmap_mode='r')
                    values=[]
                    for i in range(0,80,4):
                        def batch(x):
                            v=torch.from_numpy(x[i:i+4].copy()).cuda().permute(0,3,1,2).float()/255
                            return F.interpolate(v,(256,384),mode='bilinear',align_corners=False)*2-1
                        values.extend(metric(batch(pixels),batch(reference)).flatten().cpu().tolist())
                    row['lpips80']=float(np.mean(values))
                    rt.forward=timed
                data[f'{p}_{seed}_{name}']=row
                (folder/('start_video.json' if '--video' in sys.argv else 'start_chunks.json')).write_text(json.dumps(data))
                print(p,seed,name,row,flush=True)
run()
