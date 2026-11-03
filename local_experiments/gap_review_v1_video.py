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
from .review_v1.runtime import StreamingRuntime
from .gap_gain_vector64_runtime import GapGainVector64Runtime
from .rollout_suite import generate_chunk
from .stream_eval_common import native_vae,decode,save_video
from .high_sparse_experiment import prefix_quality
from .learned_cache.io import atomic,digest,disk_guard,RunLock
ROOT=Path('artifacts/review_v1_20261102/video')

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--method',choices=['dense','accepted','candidate'],required=True);ap.add_argument('--checkpoint');ap.add_argument('--chunks',type=int,choices=[9,41],default=41);a=ap.parse_args()
    root=ROOT/f'{a.case}_s{a.seed}'/a.method;lock=RunLock(root);rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity=dict(case=a.case,seed=a.seed,method=a.method,source=digest(__file__),runtime=digest('local_experiments/review_v1/runtime.py'),checkpoint=digest(a.checkpoint) if a.checkpoint else digest('artifacts/four_gaps_20261031/shared/epoch1.pt'),resolution=[512,768],history_pages=338,layer0_history_pages=360,page_tokens=64,probes=16,clean='dense',speed_valid=False)
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/f'result_{a.chunks}.json').exists():return
        engine=Engine();engine.wrapper.requires_grad_(False)
        if a.method=='accepted':
            rt=GapGainVector64Runtime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
            rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        else:
            m=Model('artifacts/four_gaps_20261031/shared/epoch1.pt')
            if a.method=='candidate':
                assert a.checkpoint;m.load_state_dict(torch.load(a.checkpoint,map_location='cuda',weights_only=False)['model'])
            rt=StreamingRuntime(engine,m);rt.sparse=a.method=='candidate'
        emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda();latents=[]
        for chunk in range(a.chunks):
            disk_guard(ROOT,50,2);path=root/f'chunk{chunk:02d}.pt'
            atomic(root/'status.json',dict(status='generating_or_replaying',chunk=chunk,total=a.chunks,pid=os.getpid(),updated=time.time(),speed_valid=False))
            if path.exists():
                clean=torch.load(path,map_location='cuda',weights_only=True);engine.forward(clean,emb,chunk,4)
            else:
                clean=generate_chunk(engine,emb,a.seed,chunk);assert torch.isfinite(clean).all()
                atomic(path,clean.cpu(),True)
            latents.append(clean.cpu());print(json.dumps(dict(case=a.case,method=a.method,chunk=chunk)),flush=True)
        storage=rt.finish();rt.close();rt=None;del engine;gc.collect();torch.cuda.empty_cache()
        z=torch.cat(latents,1);limit=105 if a.chunks==9 else 480
        vae=native_vae();frames=decode(vae,z.cuda(),limit)
        save_video(root/f'video_{a.chunks}.mp4',frames)
        metrics={}
        if a.method!='dense':
            import lpips
            os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
            metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
            refroot=root.parent/'dense';refs=[torch.load(refroot/f'chunk{c:02d}.pt',weights_only=True) for c in range(a.chunks)]
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
