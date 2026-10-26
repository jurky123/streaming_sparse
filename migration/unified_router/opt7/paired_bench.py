"""Same GPU, balanced alternating full-generation timings, with stable outputs."""
import argparse,gc,json,time,statistics
from functools import partial
from pathlib import Path
import torch
from ..common import config,atomic_json,update,paired_seed
from ..opt6_guarded import load_guarded_head,GuardedHeadRoute
from ..opt4.network import CompactCache
from ..fast_network import FastNetwork
from .. import fast_network
from ..opt2.attention import attention as old_attention
from .split_attention import attention
from .fused_network import FusedNetwork
from .projection_cache import HistoryProjection
from .compact_score import CompactScore
from ..speed_profile import digest,gpu_info
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--repeats',type=int,default=6);p.add_argument('--prompt-index',type=int,default=0);p.add_argument('--two',action='store_true');a=p.parse_args();root,m=config(a.directory);out=root/('optimization7/paired_val/prompt'+str(a.prompt_index)if a.two else 'optimization7/paired');out.mkdir(exist_ok=True,parents=True);paired_seed(0)
 def status(stage,**kw):update(out/'status.json',status=stage,**kw);print(stage,kw,flush=True)
 status('loading');net,_=load_guarded_head(root);base_prepare=net.prepare;plans={}
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from utils.wan_wrapper import WanVAEWrapper
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=ReplayDataset('val',require_complete=True);pid=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids'][a.prompt_index]
 sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 names=['Dense','Opt7-Reuse0']if a.two else ['Dense','Opt6','Opt7-Full','Opt7-Score','Opt7-Reuse0'];reference={};rows=[];versions=tuple(p._version for p in net.parameters())
 def configure(name):
  full=name in ['Dense','Opt6','Opt7-Full'];old=name in ['Dense','Opt6'];policy='0'if name=='Opt7-Reuse0'else'0123'
  if name not in plans:
   route=GuardedHeadRoute(batch=net.route_batch)if full else CompactScore()
   plans[name]=(route,CompactCache(route,net.route_batch),base_prepare if old else HistoryProjection(base_prepare))
  net.__class__=FastNetwork if old else FusedNetwork
  net.fast_route,net.graph_route,net.prepare=plans[name]
  net.reuse_policy=policy;net.refresh_steps=tuple(map(int,policy));net.route_cache={}
  fast_network.sparse_attention=partial(old_attention,warps=4,stages=3)if old else partial(attention,splits=2,warps=4,stages=2)
  mode='Dense'if name=='Dense'else'K32-Set'if full else'U-Sol';net.set_mode(mode);return mode
 with sdpa_backend():
  for round_id in range(a.repeats+1):
   order=names if round_id%2==0 else list(reversed(names))
   for name in order:
    mode=configure(name);gc.collect();torch.cuda.synchronize();status('warmup'if round_id==0 else'timing',round=round_id,variant=name)
    start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,mode!='Dense');torch.cuda.synchronize();generated=time.perf_counter();latent=torch.cat(chunks,1)
    pixels=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();torch.cuda.synchronize();end=time.perf_counter()
    checksum=digest(latent)
    if round_id==0:reference[name]=checksum
    else:assert checksum==reference[name]
    row=dict(round=round_id,variant=name,generation_seconds=generated-start,decode_seconds=end-generated,total_seconds=end-start,latent_sha256=checksum,gpu=gpu_info());rows.append(row)
    atomic_json(out/'timings.json',dict(prompt=pid,records=rows,repeats=a.repeats,same_gpu=True,balanced_order=True,includes_context_rebuild=True))
    print(name,row['generation_seconds'],flush=True)
    del chunks,latent,pixels;net._eval_sketch.clear();net.route_cache.clear()
 assert tuple(p._version for p in net.parameters())==versions
 result={name:{metric:statistics.median(r[metric]for r in rows if r['round']>0 and r['variant']==name)for metric in ['generation_seconds','decode_seconds','total_seconds']}for name in names}
 atomic_json(out/'summary.json',result);status('completed',summary=result)
if __name__=='__main__':main()
