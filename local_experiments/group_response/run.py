"""Independent 178f95a8 frozen grouping experiment worker."""
import argparse,json,time,gc,os
from pathlib import Path
import torch,numpy as np
from ..broad_common import Engine,ROOT as DATA
from ..review_v1.model import Model
from ..rollout_suite import generate_chunk
from ..stream_eval_common import native_vae,decode,save_video
from ..high_sparse_experiment import prefix_quality
from ..learned_cache.io import atomic,digest
from .core import Runtime,GROUPS
from .local import Collector
ROOT=Path('/tmp/group_response_results');BASE='artifacts/four_gaps_20261031/shared/epoch1.pt';PRE='artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--level',choices=['Dense','L2','L3'],required=True);p.add_argument('--collect',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 root=ROOT/'video'/f'{a.case}_s{a.seed}'/a.group/a.level;root.mkdir(parents=True,exist_ok=True)
 if (root/'result.json').exists():return
 atomic(root/'status.json',dict(status='loading',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
 engine=Engine();engine.wrapper.requires_grad_(False);model=Model(BASE);model.load_state_dict(torch.load(PRE,weights_only=False,map_location='cuda')['model']);model.requires_grad_(False)
 collector=Collector(model,ROOT,a.case,a.seed) if a.collect else None
 if collector:collector.engine=engine
 rt=Runtime(engine,model,a.group,a.level,collector);emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda();latents=[]
 gates={str(l):(.1*model[l].group_gate.tanh()).cpu().tolist() for l in range(1,30)};atomic(root/'gates.json',gates)
 begin=time.perf_counter()
 for chunk in range(9):
  atomic(root/'status.json',dict(status='generating',chunk=chunk,gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
  # Always own closed-loop masks. No baseline mask or future clean replay.
  clean=generate_chunk(engine,emb,a.seed,chunk);assert torch.isfinite(clean).all();latents.append(clean.cpu());print(a.case,a.seed,a.group,a.level,chunk,flush=True)
 torch.cuda.synchronize();generation=time.perf_counter()-begin;diagnostics=rt.diagnostics();z=torch.cat(latents,1);atomic(root/'latents.pt',z,True);atomic(root/'diagnostics.json',diagnostics)
 rt.close();del rt,engine,model;gc.collect();torch.cuda.empty_cache()
 vae=native_vae();frames=decode(vae,z.cuda(),105);save_video(root/'video.mp4',frames);quality={};latent_nmse=0.
 if a.level!='Dense':
  refpath=root.parent/'Dense/latents.pt';assert refpath.exists(),refpath
  ref=torch.load(refpath,weights_only=True);latent_nmse=float((z.float()-ref.float()).square().sum()/ref.float().square().sum());reference=decode(vae,ref.cuda(),105)
  os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve())
  import lpips
  metric=lpips.LPIPS(net='alex').eval().requires_grad_(False);q=prefix_quality(frames,reference,metric);quality={n:q[n] for n in ['lpips','ssim','psnr']}
 atomic(root/'result.json',dict(status='complete',case=a.case,seed=a.seed,group=a.group,level=a.level,frames=105,chunks=9,quality_window=[21,101],quality=quality,latent_nmse=latent_nmse,generation_seconds=generation,speed_valid=False,includes_identity_checks=True,weights=digest(PRE),history_pages=338,reference='same-group Dense with matching commit/eviction',local_collector=a.collect,frozen=True))
 atomic(root/'status.json',dict(status='complete'))
if __name__=='__main__':main()
