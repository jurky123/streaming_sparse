"""Measure complete-video drift from latest FP32 reference on a spare GPU."""
import argparse,os,time,json,shutil
from pathlib import Path
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--method',default='ours_latest_fast');p.add_argument('--reference',default='ours');a=p.parse_args()
source=Path(__file__).resolve().parents[1];root=Path('/tmp/streaming_sparse-0/runtime');folder=root/'artifacts/comparison_20261113'/a.case/a.method
while not (folder/'latents.pt').exists():time.sleep(2)
os.chdir(root);torch.set_num_threads(4)
from local_experiments.stream_eval_common import native_vae,decode
from local_experiments.high_sparse_experiment import prefix_quality
with torch.no_grad():
 vae=native_vae();z=torch.load(folder/'latents.pt',weights_only=True);ref=torch.load(folder.parent/a.reference/'latents.pt',weights_only=True)
 aa=np.concatenate(decode(vae,z.cuda(),480));bb=np.concatenate(decode(vae,ref.cuda(),480))
 os.environ['TORCH_HOME']=str(root/'artifacts/rollout_acceptance_20261030/torch_cache')
 import lpips
 metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
 rows={}
 for label,start,end in [('early',21,101),('middle',200,280),('late',400,480)]:
  x=np.concatenate((aa[:21],aa[start:end]));y=np.concatenate((bb[:21],bb[start:end]));d=prefix_quality([x],[y],metric)
  rows[label]={k:d[k] for k in ['lpips','ssim','psnr']}
 delta=z.float()-ref.float();result=dict(reference=a.reference,bitwise_equal=torch.equal(z,ref),latent_nmse=float(delta.square().mean()/ref.float().square().mean()),windows=rows)
 (folder/'quality_vs_reference.json').write_text(json.dumps(result,indent=2))
 target=source/'artifacts/comparison_20261113'/a.case/a.method;target.mkdir(exist_ok=True);shutil.copy2(folder/'quality_vs_reference.json',target/'quality_vs_reference.json');print(json.dumps(result),flush=True)
