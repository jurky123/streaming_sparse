"""Warmed full 30s generation+VAE, paired baseline/graph, exact output audit."""
import gc,json,os,signal,torch
from pathlib import Path
from .broad_common import Engine
from .all_sparse_runtime import AllSparseRuntime
from .all_sparse_graph_inference import GraphInferenceRuntime
from .rollout_suite import generate_chunk,timed_run,video
from .improve_accept import Telemetry
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/sparse_expansion_20261031/full_sparse_perf');TRAIN=ROOT.parent/'all_sparse_train'
@torch.no_grad()
def main():
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        assert json.loads((ROOT.parent/'graph_inference_audit/result.json').read_text())['passed']
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity=dict(model=digest(TRAIN/'epoch1.pt'),code={n:digest(Path('local_experiments')/n) for n in ['all_sparse_full_perf.py','all_sparse_runtime.py','all_sparse_inference.py','all_sparse_graph_inference.py']},case='basketball_court',seed=1001,repeats=2,checkpoint='epoch1',warmup_chunks=6)
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);emb=torch.load('artifacts/mass_variance_20261030/embeddings/basketball_court.pt',weights_only=True).cuda()
        early=torch.load(ROOT.parent/'all_sparse_video/basketball_court/epoch1.latents.pt',weights_only=True)
        for rep in range(2):
            for name,cls in [('reference',AllSparseRuntime),('graph',GraphInferenceRuntime)]:
                label=f'{name}_{rep}';dest=ROOT/(label+'.json')
                if dest.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted'));return
                disk_guard(ROOT,50,1);gc.collect();torch.cuda.empty_cache();rt=cls(engine,'artifacts/full_route_grad_20261031/epoch1.pt');rt.model.load_state_dict(torch.load(TRAIN/'epoch1.pt',map_location='cpu',weights_only=False)['model'])
                try:
                    atomic(ROOT/'status.json',dict(status='running',unit=label,pid=os.getpid()))
                    for c in range(6):warm=generate_chunk(engine,emb,1001,c)
                    vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                    tele=Telemetry();tele.start()
                    try:latents,frames,perf=timed_run(engine,rt,vae,emb,1001)
                    finally:load=tele.finish()
                    value=torch.cat(latents,1).cpu();assert torch.equal(value[:,:27],early)
                    ref=ROOT/'reference.latents.pt'
                    if ref.exists():assert torch.equal(value,torch.load(ref,weights_only=True))
                    else:atomic(ref,value,True);video(ROOT/'epoch1.mp4',frames)
                    atomic(dest,dict(performance=perf,runtime=rt.finish(),telemetry=load,full_latents_exact=True));print(json.dumps(dict(unit=label,seconds=perf['total_generation_vae_seconds'],generation=perf['generation_seconds'],exact=True)),flush=True)
                    del latents,frames,value
                finally:rt.close();del rt
        atomic(ROOT/'status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
