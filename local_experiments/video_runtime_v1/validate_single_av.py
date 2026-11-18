"""Paired 5-prompt x 4-seed validation of single-term AV at fixed routing policy."""
import argparse
import gc
import json
import os
from pathlib import Path
import numpy as np
import torch
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from ..stream_eval_common import native_vae, decode
from ..high_sparse_experiment import prefix_quality
from . import legacy_components
from .accepted_tiers import config
from .backbone_ops import install
from .deployment import configure, load_weights, make_runtime
from .weights import ROOT

PROMPTS=('kitchen_cook','forest_run','portrait_turn','reef_fish','tennis_match')
TIERS=('base','intersection','step')
CACHE=Path('/tmp/streaming_sparse-0/single_av_validation_20261118')
REPORT=ROOT/'artifacts/video_runtime_v1/exact_head_major_v1/validation'

@torch.no_grad()
def generate(tier, cases, group_direction=False):
    cfg=config('base' if tier=='dense' else tier,local_assets='/tmp/streaming_sparse-0/runtime')
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    if tier=='dense':
        patches=install(engine,cfg)
        reset=engine.reset
    else:
        model,reader,phi,basis,_=load_weights(cfg)
        rt=make_runtime(engine,model,reader,phi,basis,cfg)
        reset=rt.reset
    for prompt,seed in cases:
        emb=torch.load(ROOT/f'artifacts/e24_joint_representatives_v1/embeddings/{prompt}.pt',weights_only=True).cuda()
        folder=CACHE/f'{prompt}_{seed}'
        folder.mkdir(exist_ok=True,parents=True)
        modes = ('dense',) if tier=='dense' else ('old','group') if group_direction else ('old','new')
        for mode in modes:
            file=folder/f'{tier}_{mode}.pt'
            if file.exists():continue
            legacy_components.SINGLE_AV=mode=='new'
            from . import grouping_dataflow
            grouping_dataflow.FUSED_SMALL_DIRECTION=mode=='group'
            if mode=='group':
                # An existing graph already contains the original grouping kernels.
                if not getattr(rt.group_graph,'direction_variant',False):
                    rt.group_graph=grouping_dataflow.GraphGrouping(rt.config,spatial_softmax=True)
                    rt.group_graph.direction_variant=True
            reset()
            z=torch.cat([generate_chunk(engine,emb,seed,c) for c in range(9)],1).cpu()
            assert torch.isfinite(z).all()
            temp=file.with_suffix('.tmp')
            torch.save(z,temp);temp.replace(file)
    if tier=='dense':
        for patch in reversed(list(patches.values())):patch.close()
    else:rt.close()

def metrics(x,y,metric):
    return {k:v for k,v in prefix_quality([x],[y],metric).items() if k!='per_frame'}

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,required=True)
    parser.add_argument('--group-direction',action='store_true')
    args=parser.parse_args()
    report=REPORT if not args.group_direction else REPORT.parent.parent/'group_direction_v1/validation'
    cases=[(p,s) for p in PROMPTS for s in range(703,707)][args.shard::8]
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    configure(config('base',local_assets='/tmp/streaming_sparse-0/runtime'))
    for tier in ('dense',*TIERS):
        generate(tier,cases,args.group_direction)
        gc.collect();torch.cuda.empty_cache()
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    import lpips
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    vae=native_vae()
    report.mkdir(exist_ok=True)
    for prompt,seed in cases:
        folder=CACHE/f'{prompt}_{seed}'
        def frames(z):return np.concatenate(decode(vae,z.cuda(),105))
        dense=frames(torch.load(folder/'dense_dense.pt',weights_only=True))
        rows=[]
        for tier in TIERS:
            a,b=(torch.load(folder/f'{tier}_{m}.pt',weights_only=True) for m in ('old','group' if args.group_direction else 'new'))
            x,y=frames(a),frames(b)
            rows.append(dict(prompt=prompt,seed=seed,tier=tier,gpu=args.shard,old=metrics(x,dense,metric),
                             new=metrics(y,dense,metric),paired=metrics(y,x,metric),
                             nmse=((b.float()-a.float()).square().sum()/a.float().square().sum()).item()))
        (report/f'{prompt}_{seed}.json').write_text(json.dumps(rows,indent=2))
        print(prompt,seed,flush=True)

if __name__=='__main__':main()
