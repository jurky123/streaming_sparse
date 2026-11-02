"""Instrumented nine-chunk stage breakdown; not end-to-end acceptance timing."""
import json,gc
from pathlib import Path
import torch
from .broad_common import Engine
from .high_sparse_learn_runtime import LearnedHighRuntime
from .rollout_suite import generate_chunk
from .improve_accept import Telemetry
from .learned_cache.io import atomic
ROOT=Path('artifacts/high_sparse_train_20261030')
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    epoch=json.loads((ROOT/'selection.json').read_text())['selected_epoch'];cp=ROOT/f'epoch{epoch}.pt';engine=Engine()
    emb=torch.load('artifacts/mass_variance_20261030/embeddings/basketball_court.pt',weights_only=True).cuda()
    for mode in ['baseline','joint','swap']:
        dest=ROOT/'profile'/(mode+'.json')
        if dest.exists():continue
        rt=LearnedHighRuntime(engine,cp,mode)
        try:
            for c in range(6):generate_chunk(engine,emb,1001,c)
            rt.reset();rt.profile=True;tele=Telemetry();tele.start()
            try:
                for c in range(9):generate_chunk(engine,emb,1001,c)
                result=rt.finish()
            finally:load=tele.finish()
            result.update(scope='Instrumented nine chunks, CUDA event stage times; excludes VAE and uninstrumented model work; not acceptance speed',telemetry=load)
            atomic(dest,result);print(json.dumps(dict(mode=mode,times=result['profile_gpu_ms'],sparse_calls=result['sparse_calls'])),flush=True)
        finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
