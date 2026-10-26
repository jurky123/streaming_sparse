"""Warm, synchronized speed measurements; profiler runs strictly afterwards."""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch

from .common import ROOT,config,atomic_json,paired_seed,sha,update

ASSIGNMENTS=['Dense','U-Sol','K32-Set','V32-Set','V32-Set','K32-Set','U-Sol','Dense']


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def gpu_info():
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    query='index,uuid,name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu'
    return subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu='+query,'--format=csv,noheader'],text=True).strip()


def prepare(root,out,repeats):
    assert not out.exists(),'Use a new output directory to preserve measurements'
    out.mkdir(parents=True)
    e2e=json.loads((root/'e2e_manifest.json').read_text())
    identity=dict(directory=str(root),created_at=datetime.now(timezone.utc).isoformat(),
        repetitions=repeats,warmups=1,assignments=ASSIGNMENTS,prompt_id=e2e['prompt_ids'][0],generation_seed=0,
        sparse_layers=list(range(1,30)),dense_layers=[0],frames=45,resolution=[480,832],
        dtype='BF16 backbone/attention; FP32 router with TF32 disabled',
        conditioning='Existing cached prompt embedding; text encoding and model loading excluded',
        timed='Four-chunk closed-loop generation, Dense context rebuilds, VAE decode, output RGB transfer; no LPIPS or MP4 encoding',
        benchmark_instrumentation='No observer/profiler patches; synchronize before/after generation and decode',
        profiler_scope='Layer24, fourth chunk, last denoising step: CPU/CUDA operators and module ranges; separate pass',
        weight_files=e2e['frozen_files'],runtime_code=e2e['code_sha256'])
    atomic_json(out/'manifest.json',identity)


@torch.no_grad()
def worker(root,out):
    from .network import load_network
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.v08.rollout import trajectory
    from streaming_sparse.capture.capture_runner import sdpa_backend
    from utils.wan_wrapper import WanVAEWrapper
    rank=int(os.environ.get('RANK',0));mode=ASSIGNMENTS[rank];replica=0 if rank<4 else 1
    torch.cuda.set_device(0);paired_seed(0)
    m=json.loads((out/'manifest.json').read_text());work=out/f'rank{rank}';work.mkdir(exist_ok=True)
    status_path=work/'status.json'
    def status(**kw):
        update(status_path,rank=rank,mode=mode,replica=replica,pid=os.getpid(),**kw)
        print(json.dumps(kw),flush=True)
    status(status='loading')
    for path,d in {**m['weight_files'],**m['runtime_code']}.items():assert sha(path)==d,path
    started=time.perf_counter();net,_=load_network(root)
    vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    versions=tuple(p._version for p in net.parameters());load_seconds=time.perf_counter()-started
    ds=ReplayDataset('val',require_complete=True)
    sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==m['prompt_id']and r['generation_seed']==0))
    embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16);pid=m['prompt_id']
    samples=[];warmup=None;reference_latent=None
    with sdpa_backend():
        for iteration in range(m['repetitions']+1):
            net.set_mode(mode);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            status(status='warmup' if iteration==0 else 'benchmark',iteration=iteration,total=m['repetitions'])
            info=gpu_info();torch.cuda.synchronize();t0=time.perf_counter()
            chunks,diagnostics=trajectory(net,embeds,pid,0,mode!='Dense')
            torch.cuda.synchronize();t1=time.perf_counter()
            latent=torch.cat(chunks,1)
            pixels=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu()
            vae.model.clear_cache();torch.cuda.synchronize();t2=time.perf_counter()
            assert pixels.shape==(45,3,480,832)
            row=dict(iteration=iteration,mode=mode,replica=replica,gpu=info,
                generation_seconds=t1-t0,decode_and_transfer_seconds=t2-t1,total_seconds=t2-t0,
                decoded_fps=45/(t2-t0),generation_equivalent_rgb_fps=45/(t1-t0),
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                latent_sha256=digest(latent),first_chunk_sha256=digest(chunks[0]))
            if iteration==0:warmup=row;reference_latent=latent.clone()
            else:
                assert torch.equal(latent,reference_latent),'Repeated generation changed'
                samples.append(row)
            atomic_json(work/'timings.json',dict(mode=mode,replica=replica,warmup=warmup,samples=samples,
                torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),load_seconds=load_seconds))
            print(json.dumps(row),flush=True)
            del chunks,pixels,latent;gc.collect()
        atomic_json(work/'benchmark_complete.json',dict(passed=True,mode=mode,samples=len(samples)))
        status(status='waiting_for_benchmarks',samples=len(samples))
        deadline=time.monotonic()+7200
        while not all((out/f'rank{i}/benchmark_complete.json').exists()for i in range(8)):
            if time.monotonic()>deadline:raise TimeoutError('Benchmark barrier')
            time.sleep(2)
        if replica==0:
            status(status='profiling')
            # Import/patch only after EVERY uninstrumented timing trial ended.
            from .profile_observer import Observer
            observer=Observer(net,work)
            with observer:
                net.set_mode(mode)
                chunks,_=trajectory(net,embeds,pid,0,mode!='Dense')
            profiled=torch.cat(chunks,1)
            torch.testing.assert_close(profiled,reference_latent,atol=0,rtol=0)
            observer.export()
            atomic_json(work/'observer_validation.json',dict(bit_identical_latents=True,
                latent_sha256=digest(profiled),reference_sha256=digest(reference_latent)))
        assert tuple(p._version for p in net.parameters())==versions
        status(status='completed',samples=len(samples),backbone_and_router_unchanged=True)


def run(root,out,repeats):
    prepare(root,out,repeats)
    update(out/'status.json',status='running',stage='speed_benchmark',driver_pid=os.getpid())
    command=[str(ROOT/'streaming_sparse/.venv/bin/python'),'-u',str(ROOT/'migration/runtime/launch.py'),
        '--gpus','0,1,2,3,4,5,6,7','--module','migration.unified_router.speed_profile',
        '--logdir',str(out/'workers'),'--','--directory',str(root),'--output',str(out),'--worker']
    with(out/'launcher.log').open('w')as f:
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        update(out/'status.json',status='failed',stage='workers',error='See workers/rank*.log')
        raise RuntimeError('Speed/profile worker failed')
    update(out/'status.json',status='running',stage='merging')
    from .speed_report import report
    report(root,out)
    update(out/'status.json',status='completed',stage='all_finished',report=str(out/'REPORT.md'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',type=Path)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root,_=config(a.directory);out=(a.output or root/'speed_profile').resolve()
    (worker if a.worker else lambda r,o:run(r,o,a.repeats))(root,out)


if __name__=='__main__':main()
