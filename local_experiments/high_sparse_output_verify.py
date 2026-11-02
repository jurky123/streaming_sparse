"""Reuse prior quality only after exact latent identity of calibrated policies."""
import gc,json
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .high_sparse_output_runtime import OutputRuntime
from .learned_cache.io import atomic,RunLock
ROOT=Path('artifacts/high_sparse_output_20261031');OLD=Path('artifacts/high_sparse_gain_20261031/acceptance/early')
@torch.no_grad()
def main():
    assert json.loads((ROOT/'selection.json').read_text())['policy']=='all'
    lock=RunLock(ROOT/'verification')
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        engine=Engine()
        for case in ['basketball_court','butterfly_garden']:
            for policy,oldname in [('off','0.0'),('all','1.0')]:
                path=ROOT/'acceptance'/'early'/case/(policy+'.json')
                if path.exists():continue
                emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
                rt=OutputRuntime(engine,'artifacts/high_sparse_gain_20261031/model.pt',policy)
                try:
                    actual=torch.cat([generate_chunk(engine,emb,1001,c).cpu() for c in range(9)],1)
                    prior=torch.load(OLD/case/(oldname+'.latents.pt'),weights_only=True);assert torch.equal(actual,prior),(case,policy)
                    old=json.loads((OLD/case/(oldname+'.json')).read_text())
                    atomic(path,dict(case=case,mode=policy,quality=old['quality'],runtime=rt.finish(),latents_bit_exact=True,quality_reused_from=str(OLD/case/(oldname+'.json')),video_reused_from=str(OLD/case/(oldname+'.mp4')),note='Fresh rollout verifies identical latents; unchanged VAE and metric; existing video/quality reused, not a new quality measurement.'))
                    print(json.dumps(dict(case=case,policy=policy,latents_bit_exact=True)),flush=True)
                finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
        atomic(ROOT/'verification'/'status.json',dict(status='completed',verified_videos=4))
    finally:lock.close()
if __name__=='__main__':main()
