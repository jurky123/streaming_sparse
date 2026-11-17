"""Same resident model/weights/noise: old C+Risk runtime versus optimized C+Risk."""
import argparse,json,statistics,time
from pathlib import Path
import torch
from .weights import load,ROOT
from .runtime import VideoRuntime
from ..review_v1.runtime import StreamingRuntime
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from ..learned_cache.io import atomic,digest

@torch.no_grad()
def run(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,reader,phi,B,identity=load();engine=Engine();engine.wrapper.requires_grad_(False)
    emb=torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{args.case}.pt',weights_only=True).cuda();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[];latents={}
    for method in ('reference','optimized'):
        rt=StreamingRuntime(engine,model) if method=='reference' else VideoRuntime(engine,model,reader,phi,B,full_pset=False,tile_batch=6)
        def generate(chunks=None):
            rt.reset();zs=[]
            for chunk in range(args.chunks if chunks is None else chunks):zs.append(generate_chunk(engine,emb,args.seed,chunk))
            return torch.cat(zs,1)
        try:
            torch.cuda.synchronize();start=time.perf_counter();warm=generate(args.warm_chunks);torch.cuda.synchronize();warm_seconds=time.perf_counter()-start;del warm
            samples=[]
            for rep in range(args.reps):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();z=generate();torch.cuda.synchronize();seconds=time.perf_counter()-start
                if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite latent')
                samples.append(dict(seconds=seconds,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
                print(method,rep,round(seconds,3),flush=True)
            latents[method]=z.cpu();atomic(out/f'{method}_latents.pt',latents[method],True)
            rows.append(dict(method=method,warm_seconds=warm_seconds,warm_chunks=args.warm_chunks,repetitions=samples,median_seconds=statistics.median(s['seconds'] for s in samples)))
            atomic(out/'progress.json',dict(rows=rows))
        finally:rt.close();del rt
    d=latents['optimized'].float()-latents['reference'].float();nmse=float(d.square().sum()/latents['reference'].float().square().sum())
    result=dict(case=args.case,seed=args.seed,chunks=args.chunks,weights=identity,rows=rows,speedup=rows[0]['median_seconds']/rows[1]['median_seconds'],latent_nmse=nmse,latent_max_abs=float(d.abs().max()),
                scope='C+Risk old StreamingRuntime versus new VideoRuntime; same resident engine, same frozen placeholder weights/noise; matched short warmup followed by complete generation, no VAE or file IO inside timing; optimized first timed run may capture remaining page-count graph shapes',
                source={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    atomic(out/'result.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/video_runtime_v1/video_benchmark');p.add_argument('--case',default='city_walk');p.add_argument('--seed',type=int,default=701);p.add_argument('--chunks',type=int,choices=[9,41],default=9);p.add_argument('--warm-chunks',type=int,default=3);p.add_argument('--reps',type=int,default=1);run(p.parse_args())
