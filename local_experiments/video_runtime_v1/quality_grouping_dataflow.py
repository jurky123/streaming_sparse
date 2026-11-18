"""Short-video implementation differences; compare to unoptimized same-tier output."""
import argparse,json,os
import numpy as np
import torch
from .weights import ROOT
from ..stream_eval_common import native_vae,decode,save_video
from ..high_sparse_experiment import prefix_quality
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tier',choices=['base','intersection','step'],required=True);a=p.parse_args();torch.set_num_threads(4)
 os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
 import lpips
 metric=lpips.LPIPS(net='alex').eval().requires_grad_(False);vae=native_vae()
 root=ROOT/'artifacts/video_runtime_v1/grouping_dataflow_v1/video';pixels={}
 for mode in ('old','layout'):
  folder=root/(a.tier+'_'+mode);z=torch.load(folder/'latents.pt',weights_only=True).cuda();frames=decode(vae,z,105);pixels[mode]=np.concatenate(frames);save_video(folder/'video.mp4',frames)
 x=pixels['layout'];y=pixels['old'];delta=x.astype(np.float32)-y.astype(np.float32)
 result=dict(tier=a.tier,comparison='layout vs original implementation, same inference budget',rgb_equal=np.array_equal(x,y),rgb_mae_255=float(np.abs(delta).mean()),rgb_rmse_255=float(np.sqrt(np.square(delta).mean())),rgb_max_255=float(np.abs(delta).max()),quality=prefix_quality([x],[y],metric))
 (root/(a.tier+'_layout_quality.json')).write_text(json.dumps(result,indent=2)+'\n')
 save_video(root/(a.tier+'_comparison.mp4'),[np.concatenate([y,x],axis=2)])
 print({k:v for k,v in result.items() if k!='quality'}, {k:v for k,v in result['quality'].items() if k!='per_frame'})
if __name__=='__main__':main()
