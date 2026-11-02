"""Original passing compensation control, same GPU session configuration, full30s."""
import gc,json,signal
from pathlib import Path
import torch
from .broad_common import Engine
from .compensation_stable_runtime import StableCompensationRuntime
from .rollout_suite import generate_chunk,timed_run
from .improve_accept import Telemetry
from .learned_cache.io import atomic,RunLock,digest
ROOT=Path('artifacts/high_sparse_20261030')
@torch.no_grad()
def main():
    lock=RunLock(ROOT/'control');stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        for case in ['basketball_court','butterfly_garden']:
            path=ROOT/'full'/case/'stable_control.json'
            if path.exists():continue
            if stop[0]:return
            emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda();rt=StableCompensationRuntime(engine,'stable',(64,16,4,1,False))
            try:
                for c in range(6):warm=generate_chunk(engine,emb,1001,c)
                vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                telemetry=Telemetry();telemetry.start()
                try:latents,frames,result=timed_run(engine,rt,vae,emb,1001)
                finally:load=telemetry.finish()
                old=Path('artifacts/compensation_probe_20261030/acceptance')/(case+'_s1001')/'rank8_raw.latents.pt'
                assert torch.equal(torch.cat(latents,1).cpu(),torch.load(old,weights_only=True))
                result.update(case=case,method='original_stable_compensation',telemetry=load,source_sha=digest(__file__),same_full_latents=True)
                atomic(path,result);print(json.dumps(dict(case=case,seconds=result['total_generation_vae_seconds'],generation_seconds=result['generation_seconds'],clean=load['no_observed_foreign_processes'])),flush=True)
                del latents,frames
            finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
    finally:lock.close()
if __name__=='__main__':main()
