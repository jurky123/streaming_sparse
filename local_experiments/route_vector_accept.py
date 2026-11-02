"""Matched all-layer trained router, two scenes x two orders, exact full latent acceptance."""
import gc,json,os,signal
from pathlib import Path
import torch
from .broad_common import Engine
from .route_graph_runtime import GraphRuntime
from .route_vector_runtime import VectorRuntime as WholeRuntime
from .rollout_suite import generate_chunk,timed_run
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/route_kernel_20261031/vector');MODEL='artifacts/full_route_grad_20261031/epoch1.pt'
@torch.no_grad()
def main():
    lock=RunLock(ROOT/'acceptance');stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT/'checks.json').read_text())['passed']
        files=['route_vector_accept.py','route_vector_runtime.py','route_vector_features.py','route_whole_runtime.py','route_norm_runtime.py','route_norm_kernels.py','route_fused_kernels.py','route_packed_runtime.py','route_fused_runtime.py','route_graph_runtime.py','high_sparse_gain_runtime.py','high_sparse_learn_kernels.py','high_sparse_gain.py','high_sparse_learn.py','broad_common.py','high_sparse_runtime.py','compensation_stable_kernels.py','rollout_runtime.py','rollout_suite.py']
        identity=dict(code={p:digest(Path('local_experiments')/p) for p in files},model=digest(MODEL),repeats=2,cases=['basketball_court','butterfly_garden'],seed=1001)
        ip=ROOT/'acceptance/identity.json'
        if ip.exists():assert json.loads(ip.read_text())==identity
        else:atomic(ip,identity)
        snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
        import shutil
        for p in files:
            if not (snapshot/p).exists():shutil.copy2(Path('local_experiments')/p,snapshot/p)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        for case in identity['cases']:
            folder=ROOT/'acceptance'/case;folder.mkdir(parents=True,exist_ok=True)
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            prior=torch.load(Path('artifacts/full_route_grad_20261031/acceptance/early')/case/'trained.latents.pt',weights_only=True)
            fullref=folder/'graph.latents.pt'
            for rep in range(2):
                for method in (['graph','whole'] if rep==0 else ['whole','graph']):
                    dest=folder/f'{method}.rep{rep}.json'
                    if dest.exists():continue
                    if stop[0]:atomic(ROOT/'acceptance/status.json',dict(status='interrupted'));return
                    disk_guard(ROOT,50,1)
                    # Remove previous implementation's graph/workspace before measuring physical peaks.
                    for attr in ['shared_route_graph','shared_vector_router']:
                        if hasattr(engine,attr):delattr(engine,attr)
                    gc.collect();torch.cuda.empty_cache()
                    rt=(GraphRuntime if method=='graph' else WholeRuntime)(engine,MODEL,1.)
                    try:
                        atomic(ROOT/'acceptance/status.json',dict(status='running',case=case,rep=rep,method=method,pid=os.getpid()))
                        warm=torch.cat([generate_chunk(engine,emb,1001,c) for c in range(6)],1)
                        vae.model.clear_cache();vae.decode_to_pixel(warm[:,-3:],use_cache=True);vae.decode_to_pixel(warm[:,-3:],use_cache=True);del warm
                        telemetry=Telemetry();telemetry.start()
                        try:latents,frames,perf=timed_run(engine,rt,vae,emb,1001)
                        finally:load=telemetry.finish()
                        value=torch.cat(latents,1).cpu();assert torch.equal(value[:,:27],prior),(case,method,'prefix differs')
                        if method=='graph' and rep==0:atomic(fullref,value,True)
                        else:assert torch.equal(value,torch.load(fullref,weights_only=True)),(case,method,rep,'full latent differs')
                        atomic(dest,dict(case=case,method=method,rep=rep,prefix_bit_exact=True,full_bit_exact=True,performance=perf,telemetry=load,runtime=rt.finish(),note='41chunks/480frames. generation+VAE; excludes capture/warmup/encoding. No RGB-quality claim beyond previously evaluated matching trajectory.'))
                        print(json.dumps(dict(case=case,method=method,rep=rep,seconds=perf['total_generation_vae_seconds'],clean=load['no_observed_foreign_processes'],peak_allocated=perf['peak_allocated_bytes'])),flush=True)
                        del latents,frames,value
                    finally:rt.close();del rt
        atomic(ROOT/'acceptance/status.json',dict(status='complete',units=8))
    except Exception as exc:atomic(ROOT/'acceptance/status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
