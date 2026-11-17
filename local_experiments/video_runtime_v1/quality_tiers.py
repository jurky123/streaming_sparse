"""Implementation errors and the independent last-step inference budget, short video only."""
import argparse,json,os,subprocess
import numpy as np
import torch
from .weights import ROOT
from ..high_sparse_experiment import prefix_quality

def main():
 p=argparse.ArgumentParser();p.add_argument('--comparison',action='store_true');p.add_argument('--names',nargs='+',default=['full_m16','fast_m16']);a=p.parse_args();torch.set_num_threads(4)
 root=ROOT/'artifacts/video_runtime_v1/tiers_m16_v1';old=ROOT/'artifacts/video_runtime_v1/quality_batch_9d4d33d_v1/videos/city_walk_s701/chunks9'
 os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
 import lpips
 metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
 result={}
 for name in a.names:
  control='full_m16' if 'g2' in name else ('fast_q64' if name.startswith('fast') else 'full_q64')
  x=np.load(root/name/'pixels.npy',mmap_mode='r');y=np.load(root/control/'pixels.npy',mmap_mode='r')
  za=torch.load(root/name/'latents.pt',weights_only=True);zb=torch.load(root/control/'latents.pt',weights_only=True)
  delta=x.astype(np.float32)-y.astype(np.float32)
  result[name]=dict(control=control,latent_equal=torch.equal(za,zb),latent_nmse=float((za.double()-zb.double()).square().sum()/zb.double().square().sum()),rgb_equal=np.array_equal(x,y),rgb_mae_255=float(np.abs(delta).mean()),rgb_max_255=float(np.abs(delta).max()),vs_control=prefix_quality([x],[y],metric))
  if name.startswith('last') or name in ('full_m16','fast_m16'):
   dense=np.load(old/'dense/reference_pixels.npy',mmap_mode='r');result[name]['vs_dense']=prefix_quality([x],[dense],metric)
  (root/(name+'_quality.json')).write_text(json.dumps(result[name],indent=2))
  print(name,{k:v for k,v in result[name].items() if not isinstance(v,dict)},flush=True)
 continuity={}
 for name,method in [('full_q64','c_pset'),('fast_q64','c_risk')]:
  new=torch.load(root/name/'latents.pt',weights_only=True);ref=torch.load(old/method/'latents.pt',weights_only=True)
  continuity[name]=torch.equal(new,ref);print('continuity',name,continuity[name],flush=True)
 (root/'continuity.json').write_text(json.dumps(continuity,indent=2))
 if a.comparison:
  inputs=[old/'dense/video.mp4',*(root/n/'video.mp4' for n in ['fast_m16','last_m16','full_m16'])]
  cmd=['ffmpeg','-y','-loglevel','error']
  for path in inputs:cmd+=['-i',str(path)]
  subprocess.run(cmd+['-filter_complex','hstack=inputs=4','-an','-c:v','libx264','-threads','4','-crf','20',str(root/'budget_comparison.mp4')],check=True)
if __name__=='__main__':main()
