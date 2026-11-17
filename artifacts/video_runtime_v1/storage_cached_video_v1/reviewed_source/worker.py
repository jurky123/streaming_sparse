"""One resident worker per GPU; generation and VAE/encoding/metrics are separate stages."""
import argparse,json,os,socket,time
from pathlib import Path
import numpy as np
import torch
from .queue import Queue,folder
from .weights import ROOT
from ..learned_cache.io import atomic,digest,RunLock

@torch.no_grad()
def generate(queue,key,job,rt,identity,worker):
    from ..rollout_suite import generate_chunk
    out=folder(queue.root,job);out.mkdir(parents=True,exist_ok=True)
    meta=dict(job=job,weights=identity,history_pages=rt.config.history_pages,page=64,query=64,R=4,probes=16,B=16,density=.1,
              deployment=identity['deployment'],tile_batch=rt.tile_batch,code={p.name:digest(p) for p in Path(__file__).parent.glob('*.py') if p.name in ('deployment.py','runtime.py','scores.py','attention.py','page.py','pipeline_graph.py','weights.py','worker.py','reader_reduce.py','reader_split.py','endpoint_den.py','scores_moments.py','scores_packed.py','exclude_scan.py','risk_projected.py','logmass_cached.py','legacy_components.py')})
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==meta,'Generation identity changed; use a new output directory'
    else:atomic(out/'identity.json',meta)
    rt.full_pset=job['method']=='c_pset';rt.sparse=job['method']!='dense';rt.reset()
    embpath=ROOT/'artifacts/e24_joint_representatives_v1/embeddings'/f'{job["case"]}.pt'
    if not embpath.exists():embpath=ROOT/'artifacts/broad_coverage_20261030/embeddings'/f'{job["case"]}.pt'
    emb=torch.load(embpath,weights_only=True).cuda();latents=[];begin=time.perf_counter();replayed=0
    torch.cuda.reset_peak_memory_stats()
    for chunk in range(job['chunks']):
        path=out/f'chunk{chunk:02d}.pt'
        if path.exists():
            clean=torch.load(path,map_location='cuda',weights_only=True);rt.engine.forward(clean,emb,chunk,4);replayed+=1
        else:
            clean=generate_chunk(rt.engine,emb,job['noise_seed'],chunk)
            if not torch.isfinite(clean).all():raise FloatingPointError(f'Nonfinite clean latent in chunk {chunk}')
            atomic(path,clean.cpu(),True)
        latents.append(clean.cpu());queue.update(key,worker,'running')
        atomic(out/'status.json',dict(stage='generate',chunk=chunk,chunks=job['chunks'],seconds=time.perf_counter()-begin))
    stats=rt.finish()
    if stats['counters'][2]:raise FloatingPointError('Nonfinite Full_Pset scores; inspect runtime counters')
    atomic(out/'latents.pt',torch.cat(latents,1),True)
    atomic(out/'generation.json',dict(job=job,weights=identity,seconds=time.perf_counter()-begin,replayed_chunks=replayed,runtime=stats,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),speed_valid=False,
        timing_scope='generation + clean commits + latent IO; replay distinguished; module CUDA events optional'))
    print(json.dumps(dict(stage='generate',job=job,seconds=time.perf_counter()-begin)),flush=True)

@torch.no_grad()
def postprocess(queue,key,job,vae,metric,worker):
    from ..stream_eval_common import decode,save_video
    from ..high_sparse_experiment import prefix_quality
    out=folder(queue.root,job);z=torch.load(out/'latents.pt',weights_only=True).cuda();limit=105 if job['chunks']==9 else 480
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();begin=time.perf_counter()
    frames=decode(vae,z,limit);b.record();torch.cuda.synchronize();decode_ms=a.elapsed_time(b);decode_wall=time.perf_counter()-begin
    start=time.perf_counter();save_video(out/'video.mp4',frames);encode_seconds=time.perf_counter()-start
    pixels=np.concatenate(frames);quality={};side_seconds=0.
    if job['method']=='dense':
        # Dense is decoded once; retain a memory-mappable pixel cache for all methods.
        tmp=out/'reference_pixels.tmp.npy';np.save(tmp,pixels);os.replace(tmp,out/'reference_pixels.npy')
    else:
        refdir=folder(queue.root,dict(job,method='dense'));reference=np.load(refdir/'reference_pixels.npy',mmap_mode='r')
        for label,start,stop in ([('early',21,101)] if limit==105 else [('early',21,101),('middle',200,280),('late',400,480)]):
            pred=np.concatenate((pixels[:21],pixels[start:stop]));ref=np.concatenate((reference[:21],reference[start:stop]))
            quality[label]=prefix_quality([pred],[ref],metric);quality[label]['window']=[start,stop]
        start=time.perf_counter();side=[np.concatenate((reference[i:i+12],pixels[i:i+12]),axis=2) for i in range(0,limit,12)]
        save_video(out/'side_by_side.mp4',side);side_seconds=time.perf_counter()-start
    row=dict(job=job,quality=quality,VAE_decode_gpu_ms=decode_ms,VAE_decode_wall_seconds=decode_wall,video_encode_seconds=encode_seconds,side_by_side_seconds=side_seconds,
             quality_status='full-coverage baseline; paired Dense metrics and visual inspection',dense_reference_decode_reused=True)
    atomic(out/'postprocess.json',row);print(json.dumps(dict(stage='postprocess',job=job,decode_seconds=decode_wall)),flush=True)

@torch.no_grad()
def run(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from .deployment import current,configure,load_weights,make_runtime,identity as config_identity
    deployment=current(model_seed=args.model_seed,checkpoint=args.checkpoint,reader_checkpoint=args.reader_checkpoint,tile_batch=args.tile_batch,history_pages=args.history_pages,profile=args.profile,validate=args.validate,local_assets=args.local_assets)
    configure(deployment)
    q=Queue(args.root)
    if args.manifest:q.initialize(args.manifest)
    q.recover();worker=f'{socket.gethostname()}:{os.getpid()}'
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','unset');lock=RunLock(Path(args.root)/'workers'/f'gpu{gpu}')
    rt=None;vae=None;metric=None;identity=None;loaded=0
    try:
        while True:
            claim=q.claim(args.stage,worker)
            if claim is None:break
            key,job=claim
            try:
                if args.stage=='generate':
                    if rt is None:
                        from ..broad_common import Engine
                        model,reader,phi,B,identity=load_weights(deployment)
                        identity['deployment']=config_identity(deployment)
                        identity['runtime_code']={name:digest(Path(__file__).parent/name) for name in ('deployment.py','runtime.py','scores.py','attention.py','page.py','pipeline_graph.py','weights.py','worker.py','reader_reduce.py','reader_split.py','endpoint_den.py','scores_moments.py','scores_packed.py','exclude_scan.py','risk_projected.py','logmass_cached.py','legacy_components.py')}
                        q.bind_identity(dict(weights=identity,deployment=config_identity(deployment)))
                        engine=Engine();engine.wrapper.requires_grad_(False)
                        rt=make_runtime(engine,model,reader,phi,B,deployment);loaded+=1
                    generate(q,key,job,rt,identity,worker)
                else:
                    if vae is None:
                        from ..stream_eval_common import native_vae
                        import lpips
                        vae=native_vae();os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
                        metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
                    postprocess(q,key,job,vae,metric,worker)
                q.update(key,worker,'done')
            except BaseException as exc:q.update(key,worker,'failed',repr(exc));raise
            atomic(Path(args.root)/'workers'/f'{worker.replace(":","_")}.json',dict(stage=args.stage,model_loads=loaded,status=q.status()))
        print(json.dumps(dict(worker=worker,model_loads=loaded,status=q.status())),flush=True)
    finally:
        if rt is not None:rt.close()
        lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest');p.add_argument('--stage',choices=['generate','postprocess'],required=True)
    p.add_argument('--model-seed',type=int,default=17);p.add_argument('--checkpoint',default=str(ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt'));p.add_argument('--reader-checkpoint',default=str(ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt'));p.add_argument('--history-pages',type=int,default=360);p.add_argument('--local-assets');p.add_argument('--tile-batch',type=int,default=72);p.add_argument('--profile',action='store_true');p.add_argument('--validate',action='store_true');run(p.parse_args())
