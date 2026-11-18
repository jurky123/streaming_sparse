"""Decode paired outputs from the exact-AV benchmark using the project metrics."""
import argparse
import json
import os
import numpy as np
import torch
from .weights import ROOT
from ..stream_eval_common import native_vae, decode, save_video
from ..high_sparse_experiment import prefix_quality

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tier',required=True)
    parser.add_argument('--group-direction',action='store_true')
    args=parser.parse_args()
    tier=args.tier
    root=ROOT/'artifacts/video_runtime_v1'/('group_direction_v1' if args.group_direction else 'exact_head_major_v1')
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    import lpips
    torch.set_num_threads(4)
    zs=torch.load(root/f'{tier}.pt',weights_only=True)
    vae=native_vae()
    x,y=(np.concatenate(decode(vae,zs[i].cuda(),105)) for i in (1,0))
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    dense=torch.load(root.parent/'fair_dense_v1/video/gpu6_dense/latents.pt',weights_only=True)
    reference=np.concatenate(decode(vae,dense.cuda(),105))
    delta=zs[1].float()-zs[0].float()
    result=dict(nmse=(delta.square().sum()/zs[0].float().square().sum()).item(),
                quality=prefix_quality([x],[y],metric),old_vs_dense=prefix_quality([y],[reference],metric),
                new_vs_dense=prefix_quality([x],[reference],metric))
    (root/f'{tier}_quality.json').write_text(json.dumps(result,indent=2))
    save_video(root/f'{tier}_comparison.mp4',[np.concatenate((y,x),axis=2)])

if __name__=='__main__':
    main()
