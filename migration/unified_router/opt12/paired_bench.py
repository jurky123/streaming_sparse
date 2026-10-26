"""Paired timing restores both prototype/projection parameters and heads."""
import argparse,json,gc,time,statistics
from pathlib import Path
import torch
from ..common import config,atomic_json,update,paired_seed
from ..opt10.network import load as parent_load
from ..speed_profile import digest,gpu_info
from .. import fast_network
from .network import configure
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--prompt-index',type=int,default=0);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();root,m=config(a.directory);base=Path(m['parent_experiment']);out=root/f'paired/prompt{a.prompt_index}';out.mkdir(parents=True,exist_ok=True);paired_seed(0)
 def status(stage,**kw):update(out/'status.json',status=stage,**kw);print(stage,kw,flush=True)
 status('loading');net,_=parent_load(base);base_prepare=net.prepare
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from utils.wan_wrapper import WanVAEWrapper
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);ds=ReplayDataset('val',require_complete=True);pid=json.loads((base/'e2e_manifest.json').read_text())['prompt_ids'][a.prompt_index];sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
 names=['Dense','CompactBefore','CompactAfter','AggressiveBefore','AggressiveAfter'];plans={};attrs=['fast_route','graph_route','prepare','reuse_policy','refresh_steps'];records=[];torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 expected={n:torch.load(root/f'quality/{pid}.seed0/{n}.latent.pt',map_location='cpu',weights_only=True)for n in names}
 def select(name):
  if name not in plans:
   variant='Aggressive'if name.startswith('Aggressive')else'Compact';configure(net,root,variant,name.endswith('After'),base_prepare)
   plans[name]=dict(attrs=[getattr(net,k)for k in attrs],kernel=fast_network.sparse_attention,states={k:{kk:vv.detach().clone()for kk,vv in v.dictionary().items()}for k,v in net.router.states.items()},heads={k:{kk:vv.detach().clone()for kk,vv in v.state_dict().items()}for k,v in net.router.heads.items()if k.endswith('K32-Set')})
  plan=plans[name]
  for k,v in zip(attrs,plan['attrs']):setattr(net,k,v)
  for k,v in plan['heads'].items():net.router.heads[k].load_state_dict(v)
  for layer,params in plan['states'].items():
   for key,value in net.router.states[layer].dictionary().items():value.copy_(params[key])
  fast_network.sparse_attention=plan['kernel'];net.route_cache={};net._eval_sketch.clear();net.set_mode('Dense'if name=='Dense'else'K32-Set')
 with sdpa_backend():
  for round_id in range(a.repeats+1):
   for name in names if round_id%2==0 else list(reversed(names)):
    select(name);gc.collect();torch.cuda.synchronize();status('warmup'if round_id==0 else'timing',round=round_id,variant=name)
    start=time.perf_counter();chunks,_=trajectory(net,embeds,pid,0,name!='Dense');torch.cuda.synchronize();generated=time.perf_counter();z=torch.cat(chunks,1);pix=vae.decode_to_pixel(z.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();torch.cuda.synchronize();end=time.perf_counter();assert torch.equal(z,expected[name]),(pid,name,round_id)
    records.append(dict(round=round_id,variant=name,generation_seconds=generated-start,total_seconds=end-start,decode_seconds=end-generated,latent_sha256=digest(z)));atomic_json(out/'timings.json',dict(prompt_id=pid,records=records,repeats=a.repeats,same_gpu=True,gpu=gpu_info(),includes_native_context_rebuild=True));del chunks,z,pix;net._eval_sketch.clear();net.route_cache.clear()
 summary={name:{key:statistics.median(r[key]for r in records if r['round']>0 and r['variant']==name)for key in ['generation_seconds','decode_seconds','total_seconds']}for name in names};atomic_json(out/'summary.json',summary);status('completed')
if __name__=='__main__':main()
