"""Compare decoded compensation candidates to the matched native Dense video."""
import json,os,sys
from pathlib import Path
import torch
import numpy as np
from local_experiments.stream_eval_common import native_vae,decode,save_video
from local_experiments.high_sparse_experiment import prefix_quality
from local_experiments.video_runtime_v1.weights import ROOT
import lpips

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    model=lpips.LPIPS(net='alex').eval()
    vae=native_vae()
    root=Path(__file__).parent
    variant=sys.argv[1]
    pair=torch.load(f'/tmp/jzk_{variant}_pair.pt',weights_only=True)
    dense=torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt',weights_only=True)
    videos=[decode(vae,z.cuda(),105) for z in (dense,pair[0],pair[1])]
    result={}
    for name,frames in zip(('base',variant),videos[1:]):
        result[name]=prefix_quality(frames,videos[0],model)
    (root/(variant+'_quality.json')).write_text(json.dumps(result,indent=2))
    folder=Path('/mnt/data/tail_page_mask_v1')
    folder.mkdir(parents=True,exist_ok=True)
    save_video(folder/f'dense_base_{variant}.mp4',[np.concatenate(x,axis=2) for x in zip(*videos)])
if __name__=='__main__':main()
