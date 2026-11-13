"""Resumable closed-loop quality run. Shared-device times are NOT benchmarks.

Retain compact clean latents only. On restart, replay completed clean forwards
in order to reconstruct KV rather than writing multi-GB cache snapshots.
"""
import argparse,gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from .broad_common import Engine,ROOT as DATA
from .review_v1.model import Model
from .review_v1.ablation import AblationRuntime,VARIANTS
from .gap_gain_vector64_runtime import GapGainVector64Runtime
from .rollout_suite import generate_chunk
from .stream_eval_common import native_vae,decode,save_video
from .high_sparse_experiment import prefix_quality
from .learned_cache.io import atomic,digest,disk_guard,RunLock
ORIGINAL=Path('artifacts/review_v1_20261102/video')
ROOT=Path('artifacts/flow_single_20261104/video')

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--variant',choices=VARIANTS,required=True);ap.add_argument('--checkpoint');ap.add_argument('--chunks',type=int,choices=[9,41],default=41);a=ap.parse_args()
    root=ROOT/f'{a.case}_s{a.seed}'/a.variant;lock=RunLock(root);rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        base='artifacts/four_gaps_20261031/shared/epoch1.pt'
        a.checkpoint='artifacts/flow_single_20261104/train/reef_fish/model.pt'
        assert a.variant=='full'
        identity=dict(case=a.case,seed=a.seed,method=a.variant,source=digest(__file__),runtime=digest('local_experiments/review_v1/runtime.py'),ablation=digest('local_experiments/review_v1/ablation.py'),control='frozen-weight removal; mean/detail changes applied after full routing; storage/budget unchanged',checkpoint=digest(a.checkpoint) if a.checkpoint else digest('artifacts/four_gaps_20261031/shared/epoch1.pt'),resolution=[512,768],history_pages=338,layer0_history_pages=360,page_tokens=64,probes=16,clean='dense',speed_valid=False)
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/f'result_{a.chunks}.json').exists():return
        engine=Engine();engine.wrapper.requires_grad_(False)
        m=Model(base)
        if a.variant!='no_new_training':m.load_state_dict(torch.load(a.checkpoint,map_location='cuda',weights_only=False)['model'])
        old=torch.load(base,map_location='cpu',weights_only=False)['model']
        centers={l:old[f'layers.{l}.center'].cuda() for l in range(1,30)}
        rt=AblationRuntime(engine,m,a.variant,centers)
        emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda();latents=[]
        for chunk in range(a.chunks):
            disk_guard(ROOT,50,2);path=root/f'chunk{chunk:02d}.pt'
            atomic(root/'status.json',dict(status='generating_or_replaying',chunk=chunk,total=a.chunks,pid=os.getpid(),updated=time.time(),speed_valid=False))
            if path.exists():
                clean=torch.load(path,map_location='cuda',weights_only=True);engine.forward(clean,emb,chunk,4)
            else:
                clean=generate_chunk(engine,emb,a.seed,chunk);assert torch.isfinite(clean).all()
                atomic(path,clean.cpu(),True)
            latents.append(clean.cpu());print(json.dumps(dict(case=a.case,method=a.variant,chunk=chunk)),flush=True)
        storage=rt.finish();extra=sum(t.numel()*t.element_size() for t in centers.values());storage['ablation_center_bytes']=extra;storage['persistent_bytes']+=extra;storage['within_limit']=storage['persistent_bytes']<=storage['limit_bytes'];assert storage['within_limit'];rt.close();rt=None;del engine;gc.collect();torch.cuda.empty_cache()
        z=torch.cat(latents,1);limit=105 if a.chunks==9 else 480
        vae=native_vae();frames=decode(vae,z.cuda(),limit)
        save_video(root/f'video_{a.chunks}.mp4',frames)
        metrics={}
        if a.variant!='dense':
            import lpips
            os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
            metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
            refroot=ORIGINAL/f'{a.case}_s{a.seed}'/'dense';refs=[torch.load(refroot/f'chunk{c:02d}.pt',weights_only=True) for c in range(a.chunks)]
            reference=decode(vae,torch.cat(refs,1).cuda(),limit)
            for name,start,stop in ([('early',21,101)] if a.chunks==9 else [('early',21,101),('middle',200,280),('late',400,480)]):
                # Existing metric indexes [21:101]; shift other windows into that slice.
                aa=np.concatenate(frames);bb=np.concatenate(reference)
                if start!=21:
                    aa=np.concatenate((aa[:21],aa[start:stop]));bb=np.concatenate((bb[:21],bb[start:stop]))
                row=prefix_quality([aa],[bb],metric);row['window']=[start,stop];metrics[name]=row
        row=dict(identity=identity,chunks=a.chunks,frames=limit,quality=metrics,runtime=storage,speed_valid=False,status='measured_pending_comparison_and_visual_review')
        atomic(root/f'result_{a.chunks}.json',row);atomic(root/'status.json',dict(status='complete',chunks=a.chunks))
    except BaseException as e:atomic(root/'status.json',dict(status='failed',error=repr(e)));raise
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
