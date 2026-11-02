import gc,json,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .rollout_runtime import Runtime
from .rollout_suite import generate_chunk,timed_run
from .improve_accept import Telemetry
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/four_gaps_20261031')
@torch.no_grad()
def main():
    root=ROOT/'performance_dense';lock=RunLock(root)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        from utils.wan_wrapper import WanVAEWrapper
        engine=Engine();vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda();early=torch.load(ROOT/'validation/val_horse_field_s101/dense.latents.pt',weights_only=True)
        reference=None
        for rep in range(2):
            dest=root/f'dense_{rep}.json'
            if dest.exists():continue
            gc.collect();torch.cuda.empty_cache();rt=Runtime(engine,'dense')
            try:
                atomic(root/'status.json',dict(status='running',rep=rep))
                for c in range(6):warm=generate_chunk(engine,emb,101,c)
                vae.model.clear_cache();vae.decode_to_pixel(warm,use_cache=True);vae.decode_to_pixel(warm,use_cache=True);del warm
                tele=Telemetry();tele.start()
                try:zs,frames,perf=timed_run(engine,rt,vae,emb,101)
                finally:load=tele.finish()
                value=torch.cat(zs,1).cpu();assert torch.equal(value[:,:27],early)
                if reference is not None:assert torch.equal(value,reference)
                reference=value;atomic(dest,dict(performance=perf,runtime=rt.finish(),telemetry=load));print(json.dumps(dict(rep=rep,seconds=perf['total_generation_vae_seconds'])),flush=True);del zs,frames
            finally:rt.close();del rt
        atomic(root/'status.json',dict(status='complete'))
    except Exception as exc:atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
