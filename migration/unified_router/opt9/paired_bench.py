"""Balanced same-GPU comparisons after independent optimization and quality checks."""
import argparse,gc,json,time,statistics
import torch
from ..common import config,atomic_json,update,paired_seed
from ..opt6_guarded import load_guarded_head
from .. import fast_network
from .network import configure
from ..speed_profile import digest,gpu_info
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--repeats',type=int,default=6);p.add_argument('--prompt-index',type=int,default=0);p.add_argument('--two',action='store_true');a=p.parse_args();root,m=config(a.directory);out=root/('optimization9/paired_val/prompt'+str(a.prompt_index)if a.two else'optimization9/paired');out.mkdir(exist_ok=True,parents=True);paired_seed(0)
 def status(stage,**kw):update(out/'status.json',status=stage,**kw);print(stage,kw,flush=True)
 status('loading');net,_=load_guarded_head(root);base_prepare=net.prepare;plans={}
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from utils.wan_wrapper import WanVAEWrapper
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=ReplayDataset('val',require_complete=True);pid=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids'][a.prompt_index]
 sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 names=['Dense','Opt8Full','Packed','Skip2','Skip0','Opt8Fast','Skip2Fast','Skip0Fast'];reference={};rows=[];versions=tuple(p._version for p in net.parameters())
 attrs=['fast_route','graph_route','prepare','reuse_policy','refresh_steps']
 def select(name):
  if name not in plans:
   configure(net,'Opt8Full'if name=='Dense'else name,base_prepare)
   plans[name]=([getattr(net,k)for k in attrs],fast_network.sparse_attention)
  vals,kernel=plans[name]
  for k,v in zip(attrs,vals):setattr(net,k,v)
  fast_network.sparse_attention=kernel;net.route_cache={}
  mode='Dense'if name=='Dense'else'U-Sol'if name.endswith('Fast')else'K32-Set';net.set_mode(mode);return mode
 with sdpa_backend():
  for round_id in range(a.repeats+1):
   order=names if round_id%2==0 else list(reversed(names))
   for name in order:
    mode=select(name);gc.collect();torch.cuda.synchronize();status('warmup'if round_id==0 else'timing',round=round_id,variant=name)
    start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,mode!='Dense');torch.cuda.synchronize();generated=time.perf_counter();latent=torch.cat(chunks,1)
    pixels=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();torch.cuda.synchronize();end=time.perf_counter();checksum=digest(latent)
    if round_id==0:reference[name]=checksum
    else:assert checksum==reference[name]
    if name in ['Packed','Skip2','Skip0','Skip2Fast','Skip0Fast']:
     quality=torch.load(root/f'optimization9/quality/{pid}.seed0/{name}.latent.pt',map_location='cpu',weights_only=True)
     assert torch.equal(latent,quality),(pid,name)
    row=dict(round=round_id,variant=name,generation_seconds=generated-start,decode_seconds=end-generated,total_seconds=end-start,latent_sha256=checksum,gpu=gpu_info());rows.append(row)
    atomic_json(out/'timings.json',dict(prompt=pid,records=rows,repeats=a.repeats,same_gpu=True,balanced_order=True,includes_context_rebuild=True));print(name,row['generation_seconds'],flush=True)
    del chunks,latent,pixels;net._eval_sketch.clear();net.route_cache.clear()
 assert tuple(p._version for p in net.parameters())==versions
 result={name:{metric:statistics.median(r[metric]for r in rows if r['round']>0 and r['variant']==name)for metric in ['generation_seconds','decode_seconds','total_seconds']}for name in names}
 atomic_json(out/'summary.json',result);status('completed',summary=result)
if __name__=='__main__':main()
