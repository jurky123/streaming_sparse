"""Decode a same-prompt latent pair and compare both videos with the native Dense reference."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from local_experiments.stream_eval_common import native_vae,decode,save_video
from local_experiments.high_sparse_experiment import prefix_quality
from local_experiments.video_runtime_v1.weights import ROOT


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('pair',type=Path)
    parser.add_argument('dense',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'anchor_guard_named')
    args=parser.parse_args()
    assert json.loads(args.dense.with_name('result.json').read_text())['video']=='city_walk_s701'
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    import lpips
    torch.set_num_threads(4)
    pair=torch.load(args.pair,weights_only=True)
    dense=torch.load(args.dense,weights_only=True)
    vae=native_vae()
    old,new,reference=[np.concatenate(decode(vae,z.cuda(),105)) for z in (pair[0],pair[1],dense)]
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    result={name:prefix_quality([a],[b],metric) for name,a,b in [('old_new',old,new),('old_dense',old,reference),('new_dense',new,reference)]}
    folder=args.output
    (folder/'quality.json').write_text(json.dumps(result,indent=2))
    save_video(folder/'comparison.mp4',[np.concatenate((old,new),axis=2)])

if __name__=='__main__':main()
