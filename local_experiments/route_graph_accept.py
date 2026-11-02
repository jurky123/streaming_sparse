"""Bit-exact prefix and full-latent acceptance, paired 30s generation+VAE timing."""
import gc,json,os,signal
from pathlib import Path
import torch
from .broad_common import Engine
from .high_sparse_gain_runtime import GainRuntime
from .route_graph_runtime import GraphRuntime
from .rollout_suite import generate_chunk,timed_run
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/route_graph_20261031');MODEL='artifacts/high_sparse_gain_20261031/model.pt'
@torch.no_grad()
def main():
    lock=RunLock(ROOT/'acceptance');stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT/'checks.json').read_text())['passed']
        identity=dict(code={p:digest(Path('local_experiments')/p) for p in ['route_graph_runtime.py','route_graph_accept.py','high_sparse_gain_runtime.py']},model=digest(MODEL))
        ip=ROOT/'acceptance'/'identity.json'
        if ip.exists():assert json.loads(ip.read_text())==identity
        else:atomic(ip,identity)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        for case in ['basketball_court','butterfly_garden']:
            folder=ROOT/'acceptance'/case;folder.mkdir(parents=True,exist_ok=True)
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
            for method in ['eager','graph']:
                dest=folder/(method+'.json')
                if dest.exists():continue
                if stop[0]:return
                disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache()
                if method=='eager' and hasattr(engine,'shared_route_graph'):
                    del engine.shared_route_graph;gc.collect();torch.cuda.empty_cache()
                rt=(GainRuntime if method=='eager' else GraphRuntime)(engine,MODEL,1.)
                try:
                    atomic(ROOT/'acceptance'/'status.json',dict(status='running',case=case,method=method,pid=os.getpid()))
                    # Check both actual video cases against prior native hard rollout.
                    prefix=torch.cat([generate_chunk(engine,emb,1001,c).cpu() for c in range(9)],1)
                    prior=torch.load(Path('artifacts/high_sparse_gain_20261031/acceptance/early')/case/'1.0.latents.pt',weights_only=True)
                    assert torch.equal(prefix,prior),(case,method,'prefix trajectory mismatch')
                    vae.model.clear_cache();warm=prefix[:,-3:].cuda();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                    telemetry=Telemetry();telemetry.start()
                    try:latents,frames,performance=timed_run(engine,rt,vae,emb,1001)
                    finally:load=telemetry.finish()
                    value=torch.cat(latents,1).cpu();assert torch.equal(value[:,:27],prior)
                    fullref=folder/'eager.latents.pt'
                    if method=='eager':atomic(fullref,value,True)
                    else:assert torch.equal(value,torch.load(fullref,weights_only=True)),(case,'full trajectory mismatch')
                    result=dict(case=case,method=method,prefix_bit_exact=True,full_bit_exact_to_eager=method=='graph',frames=sum(map(len,frames)),performance=performance,telemetry=load,runtime=rt.finish(),note='No new video encoding; prefix quality reused only after exact identity; timed 41chunks/480frames includes VAE, routing and maintenance.')
                    atomic(dest,result);print(json.dumps(dict(case=case,method=method,seconds=performance['total_generation_vae_seconds'],clean=load['no_observed_foreign_processes'],peak_allocated=performance['peak_allocated_bytes'])),flush=True)
                    del value,latents,frames,prefix,prior
                finally:rt.close();del rt
        atomic(ROOT/'acceptance'/'status.json',dict(status='complete',full_videos=4))
    finally:lock.close()
if __name__=='__main__':main()
