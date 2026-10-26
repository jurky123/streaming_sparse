"""Paired native video quality and warm generation timing for vote thresholds."""
import os,json,gc,time,statistics
from pathlib import Path
import torch
from ..common import config,atomic_json,update,paired_seed,sha
from ..opt10.network import load as parent_load
from ..speed_profile import gpu_info
from .. import fast_network
from .network import configure
from .setup import ROOT

@torch.no_grad()
def main():
 root,m=config(ROOT);base=Path(m['parent_experiment']);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));out=root/'quality';out.mkdir(exist_ok=True);paired_seed(0)
 def status(stage,**kw):update(out/f'rank{rank}.json',status=stage,**kw);print(stage,kw,flush=True)
 locked=json.loads((root/'parent_lock.json').read_text())
 for path,digest in locked.items():assert sha(path)==digest,path
 status('loading');net,_=parent_load(base);base_prepare=net.prepare
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from migration.frozen_quality.worker import pixel_metrics,preview
 from utils.wan_wrapper import WanVAEWrapper
 import lpips
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False);ds=ReplayDataset('val',require_complete=True);manifest=json.loads((base/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint'];torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 names=['Dense','Reference','Vote2','Vote4','Vote8'];plans={};attrs=['fast_route','graph_route','prepare','reuse_policy','refresh_steps']
 def select(name):
  if name not in plans:
   configure(net,root,'Reference'if name=='Dense'else name,base_prepare)
   plans[name]=([getattr(net,k)for k in attrs],fast_network.sparse_attention)
  vals,kernel=plans[name]
  for k,v in zip(attrs,vals):setattr(net,k,v)
  fast_network.sparse_attention=kernel;net.route_cache={};net._eval_sketch.clear();net.set_mode('Dense'if name=='Dense'else'K32-Set')
 def decode(z):
  p=vae.decode_to_pixel(z.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return p
 with sdpa_backend():
  for pid in manifest['prompt_ids'][rank::world]:
   folder=out/f'{pid}.seed0';folder.mkdir(exist_ok=True);sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16);expected={};pixels={}
   for name in names:
    select(name);status('quality',prompt=pid,variant=name);chunks,_=trajectory(net,embeds,pid,0,name!='Dense');z=torch.cat(chunks,1);expected[name]=z.cpu();torch.save(z,folder/(name+'.latent.pt'));pixels[name]=decode(z)
    if name=='Dense':
     cached=torch.load(base/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)['Dense'];assert torch.equal(z,cached)
    else:
     assert torch.equal(z[:,:3],expected['Dense'][:,:3]);assert torch.equal(pixels[name][:9],pixels['Dense'][:9]);counts=net.routing_records;assert len(counts)==348
     keep=torch.stack([r['retained_H']+r['retained_C']for r in counts]).sum().item()/sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in counts)
     row=dict(variant=name,weighted_keep=keep,first_chunk_identical=True,quality_vs_dense=pixel_metrics(pixels['Dense'],pixels[name],metric))
     if name!='Reference':row['quality_vs_reference']=pixel_metrics(pixels['Reference'],pixels[name],metric)
     atomic_json(folder/(name+'.json'),row)
    del chunks,z;net._eval_sketch.clear();net.route_cache.clear();gc.collect()
   preview(folder/'Dense_Reference_Vote4_Vote8.mp4',pixels,['Dense','Reference','Vote4','Vote8']);del pixels;gc.collect();records=[]
   # Quality trajectories warm kernels; another round warms each cached plan.
   for round_id in range(4):
    for name in names if round_id%2==0 else list(reversed(names)):
     select(name);gc.collect();torch.cuda.synchronize();status('warmup'if round_id==0 else'timing',prompt=pid,round=round_id,variant=name)
     start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,name!='Dense');torch.cuda.synchronize();end=time.perf_counter();z=torch.cat(chunks,1);assert torch.equal(z,expected[name]),(pid,name,round_id)
     records.append(dict(round=round_id,variant=name,generation_seconds=end-start,repeat_identical=True));atomic_json(folder/'timings.json',dict(records=records,gpu=gpu_info(),includes_native_context_rebuild=True,excludes_vae=True));del chunks,z;net._eval_sketch.clear();net.route_cache.clear()
   atomic_json(folder/'timing_summary.json',{name:statistics.median(r['generation_seconds']for r in records if r['round']>0 and r['variant']==name)for name in names})
 for path,digest in locked.items():assert sha(path)==digest,path
 status('completed')
if __name__=='__main__':main()
