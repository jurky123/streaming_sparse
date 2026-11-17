"""Compare the paired short videos, plus the accepted baseline continuity."""
import os,json
import numpy as np
import torch
from .weights import ROOT
from ..high_sparse_experiment import prefix_quality

def main():
 torch.set_num_threads(4);root=ROOT/'artifacts/video_runtime_v1/storage_video_v1'
 a=torch.load(root/'baseline/latents.pt',weights_only=True);b=torch.load(root/'combined/latents.pt',weights_only=True);old=torch.load(ROOT/'artifacts/video_runtime_v1/logmass_video_v1/cached/latents.pt',weights_only=True)
 x=np.load(root/'baseline/pixels.npy',mmap_mode='r');y=np.load(root/'combined/pixels.npy',mmap_mode='r');diff=x.astype(np.float64)-y.astype(np.float64)
 errors=dict(baseline_matches_accepted=torch.equal(a,old),latent_equal=torch.equal(a,b),latent_nmse=float((a.double()-b.double()).square().sum()/a.double().square().sum()),rgb_equal=np.array_equal(x,y),rgb_mae_255=float(np.abs(diff).mean()),rgb_max_abs_255=float(np.abs(diff).max()))
 (root/'errors.json').write_text(json.dumps(errors,indent=2));print(errors,flush=True)
 os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
 import lpips
 metric=lpips.LPIPS(net='alex').eval().requires_grad_(False);quality=prefix_quality([y],[x],metric)
 (root/'quality.json').write_text(json.dumps(quality,indent=2));print({k:v for k,v in quality.items() if k!='per_frame'},flush=True)
if __name__=='__main__':main()
