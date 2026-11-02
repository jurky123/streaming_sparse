"""Separate closed-loop component profile, never mixed into acceptance wall times."""
import argparse,json,time
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_runtime import Runtime,ROOT
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,RunLock,digest

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);p.add_argument('--chunks',type=int,default=41);a=p.parse_args()
    if a.wait_pid:
        while Path(f'/proc/{a.wait_pid}').exists():time.sleep(15)
    root=ROOT/'profile';lock=RunLock(root)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        engine=Engine();emb=torch.load('artifacts/mass_variance_20261030/embeddings/basketball_court.pt',weights_only=True).cuda()
        for method in ['dense_grouped','mean','mass_mean','mass_var']:
            path=root/f'{method}.json'
            if path.exists():continue
            rt=Runtime(engine,method,profile=True)
            try:
                for ch in range(6):generate_chunk(engine,emb,606,ch)
                rt.reset();torch.cuda.synchronize();start=time.perf_counter()
                for ch in range(a.chunks):generate_chunk(engine,emb,606,ch)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                result=rt.finish();result.update(case='basketball_court',seed=606,chunks=a.chunks,profiled_generation_wall_s=elapsed,
                    note='CUDA events on real closed-loop rollout, layers1..29 only; instrumentation affects speed; no VAE; not acceptance timing. Grouping includes response projection, balanced partition, ordering and gathers.',
                    code={f:digest(Path('local_experiments')/f) for f in ['rollout_runtime.py','rollout_kernels.py','rollout_profile.py']})
                atomic(path,result);print(json.dumps(result),flush=True)
            finally:rt.close()
    finally:lock.close()

if __name__=='__main__':main()
