"""Opt12 versus Opt11 best: eight native Val videos with fresh paired baselines."""
import argparse,os,json,gc,time
from pathlib import Path
import torch
from ..common import config,atomic_json,update,paired_seed,sha
from ..opt10.network import load as parent_load
from .network import configure
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory);base=Path(m['parent_experiment']);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));out=root/'quality';out.mkdir(exist_ok=True);paired_seed(0)
 def status(stage,**kw):update(out/f'rank{rank}.json',status=stage,**kw);print(stage,kw,flush=True)
 locked={str(root/f'training/layer{layer:02d}/{variant}/best.pt'):sha(root/f'training/layer{layer:02d}/{variant}/best.pt')for layer in m['layers']for variant in ['Compact','Aggressive']}
 atomic_json(out/f'rank{rank}_checkpoint_hashes.json',locked)
 status('loading');net,_=parent_load(base);base_prepare=net.prepare
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from migration.frozen_quality.worker import pixel_metrics,preview
 from utils.wan_wrapper import WanVAEWrapper
 import lpips
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False);metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).cuda().eval().requires_grad_(False);ds=ReplayDataset('val',require_complete=True);manifest=json.loads((base/'e2e_manifest.json').read_text());assert ds.fingerprint()==manifest['val_fingerprint'];torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def decode(z):
  pixels=vae.decode_to_pixel(z.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return pixels
 audit=[]
 def observe(layer,start,step,q,k,v,sketch,prepared,mask,y):
  if layer not in [1,24,29]or step!=3:return
  ix=torch.tensor([0,1,7,31,32,51,52,1559,1560,3119,3120,4679],device=q.device);truth=torch.nn.functional.scaled_dot_product_attention(q[:,ix][None],k[None],v[None])[0].float();actual=y[:,ix].float();visible=mask[:,ix].repeat_interleave(sketch.costs.long(),-1);masked=torch.nn.functional.scaled_dot_product_attention(q[:,ix][None],k[None],v[None],attn_mask=visible[None])[0].float()
  audit.append(dict(layer=layer,start=start,step=step,sparse_to_dense_sse=float((actual-truth).square().sum()),dense_energy=float(truth.square().sum()),masked_implementation_nmse=float((actual-masked).square().sum()/masked.square().sum()),queries=144))
 net.inspect_callback=observe
 with sdpa_backend():
  for pid in manifest['prompt_ids'][rank::world]:
   folder=out/f'{pid}.seed0';folder.mkdir(exist_ok=True);sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16);cached_dense=torch.load(base/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)['Dense'];net.set_mode('Dense');status('evaluating',prompt=pid,variant='Dense');dense_chunks,_=trajectory(net,embeds,pid,0,False);dl=torch.cat(dense_chunks,1);assert torch.equal(dl,cached_dense),(pid,'Dense changed');torch.save(dl,folder/'Dense.latent.pt');densepix=decode(dl);del dense_chunks
   for variant in ['Compact','Aggressive']:
    pixels={}
    for tuned in [False,True]:
     name=variant+('After'if tuned else'Before');status('evaluating',prompt=pid,variant=name);configure(net,root,variant,tuned,base_prepare);net.set_mode('K32-Set');audit.clear();chunks,_=trajectory(net,embeds,pid,0,True);z=torch.cat(chunks,1);saved_audit=list(audit);assert torch.equal(z[:,:3],dl[:,:3]);pix=decode(z);assert torch.equal(pix[:9],densepix[:9]);pixels[tuned]=pix
     counts=net.routing_records;assert len(counts)==348;keep=torch.stack([r['retained_H']+r['retained_C']for r in counts]).sum().item()/sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in counts)
     # A separate identical trajectory checks cache refresh and determinism.
     net.inspect_callback=None;net.set_mode('K32-Set');repeat,_=trajectory(net,embeds,pid,0,True);assert torch.equal(z,torch.cat(repeat,1)),(pid,name,'repeat mismatch');net.inspect_callback=observe
     torch.save(z,folder/(name+'.latent.pt'));result=dict(variant=variant,trained=tuned,weighted_keep=keep,first_chunk_identical=True,repeat_identical=True,quality_vs_dense=pixel_metrics(densepix,pix,metric),attention_audit=saved_audit)
     if tuned:result['quality_vs_before']=pixel_metrics(pixels[False],pix,metric)
     if not tuned:
      prior=torch.load(Path(m['parent_opt11'])/f'quality/{pid}.seed0/{variant}After.latent.pt',map_location='cpu',weights_only=True);result['matches_opt11_after']=torch.equal(z,prior)
     atomic_json(folder/(name+'.json'),result);del chunks,z,repeat;net._eval_sketch.clear();net.route_cache.clear();gc.collect()
    preview(folder/f'Dense_{variant}_Before_After.mp4',{'Dense':densepix,'Before':pixels[False],'After':pixels[True]},['Dense','Before','After']);del pixels;gc.collect()
 for path,digest in locked.items():assert sha(path)==digest,path
 status('completed')
if __name__=='__main__':main()
