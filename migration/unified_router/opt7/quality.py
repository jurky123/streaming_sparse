"""Eight held-out Val prompts: native video quality and actual masked-AV audits."""
import argparse,gc,json,os
import torch
from ..common import config,atomic_json,update,paired_seed,sha
from .variant import load_variant
from .fused_network import FusedNetwork
from .compact_score import CompactScore
from ..opt6_guarded import GuardedHeadRoute
from ..opt4.network import CompactCache

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory)
 out=root/'optimization7/quality';out.mkdir(exist_ok=True);rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));paired_seed(0)
 manifest=json.loads((root/'e2e_manifest.json').read_text());seal=json.loads((root/'optimization6/guard_manifest.json').read_text())
 for path,digest in {**manifest['code_sha256'],**manifest['frozen_files'],**seal['code']}.items():assert sha(path)==digest,path
 def status(stage,**kw):update(out/f'rank{rank}.json',status=stage,rank=rank,**kw);print(stage,kw,flush=True)
 status('loading');os.environ['SLIMAV_VARIANT']='GuardCachedSplitCounters';net,_=load_variant(root)
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from migration.frozen_quality.worker import pixel_metrics,preview
 from utils.wan_wrapper import WanVAEWrapper
 import lpips
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
 metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).cuda().eval().requires_grad_(False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=ReplayDataset('val',require_complete=True);assert ds.fingerprint()==manifest['val_fingerprint'];versions=tuple(p._version for p in net.parameters())
 def decode(latent):
  pix=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return pix
 audits=[]
 def observe(layer,start,step,q,k,v,sketch,prepared,mask,y):
  if layer not in [1,24,29]or step!=3:return
  ids=torch.tensor([0,1,7,31,32,51,52,1559,1560,3119,3120,4679],device=q.device)
  visible=mask[:,ids].repeat_interleave(sketch.costs.long(),-1)
  truth=torch.nn.functional.scaled_dot_product_attention(q[:,ids][None],k[None],v[None],attn_mask=visible[None])[0]
  dense=torch.nn.functional.scaled_dot_product_attention(q[:,ids][None],k[None],v[None])[0]
  yy=y[:,ids];assert torch.isfinite(y).all()
  nmse=float((yy.float()-truth.float()).square().sum()/truth.float().square().sum())
  assert nmse<1.e-4,(layer,start,nmse)
  audits.append(dict(layer=layer,start=start,step=step,queries=ids.numel()*12,masked_attention_nmse=nmse,
    sparse_to_dense_sse=float((yy.float()-dense.float()).square().sum()),dense_energy=float(dense.float().square().sum())))
 net.inspect_callback=observe
 variants=[('GuardCachedSplitCounters','K32-Set','0123'),('ScoreCompactCachedSplitCounters','U-Sol','0123'),('ScoreCompactCachedReuse0SplitCounters','U-Sol','0')]
 with sdpa_backend():
  for pid in manifest['prompt_ids'][rank::world]:
   folder=out/f'{pid}.seed0';folder.mkdir(exist_ok=True)
   sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
   dense_latent=torch.load(root/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)['Dense'];dense=decode(dense_latent)
   prior_latent=torch.load(root/f'optimization6/guard_quality/{pid}.seed0/Guarded-K32.latent.pt',map_location='cpu',weights_only=True);prior=decode(prior_latent)
   result=dict(prompt_id=pid,seed=0,test_used=False,variants={},baseline_vs_dense=pixel_metrics(dense,prior,metric))
   for variant,kind,policy in variants:
    status('evaluating',prompt=pid,variant=variant);gc.collect();torch.cuda.empty_cache()
    route=CompactScore()if variant.startswith('Score')else GuardedHeadRoute(batch=net.route_batch)
    net.fast_route=route;net.graph_route=CompactCache(route,net.route_batch);net.reuse_policy=policy;net.refresh_steps=tuple(map(int,policy));net.route_cache={};net.set_mode(kind)
    audits.clear();chunks,_=trajectory(net,embeds,pid,0,True);latent=torch.cat(chunks,1);pix=decode(latent)
    assert torch.equal(latent[:,:3],dense_latent[:,:3]);assert torch.equal(pix[:9],dense[:9])
    records=net.routing_records;assert len(records)==348 and {r['layer']for r in records}==set(range(1,30))
    retained=torch.stack([r['retained_H']+r['retained_C']for r in records]).sum().item();denom=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in records)
    row=dict(variant=variant,mode=kind,refresh_steps=list(net.refresh_steps),weighted_keep=retained/denom,
      quality_vs_dense=pixel_metrics(dense,pix,metric),quality_vs_opt6=pixel_metrics(prior,pix,metric),
      latent_nmse_by_chunk=[float((latent[:,j:j+3].float()-prior_latent[:,j:j+3].float()).square().sum()/prior_latent[:,j:j+3].float().square().sum())for j in range(0,12,3)],
      first_chunk_identical=True,attention_audit=list(audits))
    atomic_json(folder/(variant+'.json'),row);torch.save(latent,folder/(variant+'.latent.pt'));result['variants'][variant]=row
    preview(folder/f'Dense_Opt6_{variant}.mp4',{'Dense':dense,'Opt6':prior,'Opt7':pix},['Dense','Opt6','Opt7'])
    print(variant,'LPIPS',row['quality_vs_dense']['full']['lpips'],'keep',row['weighted_keep'],flush=True)
    del chunks,latent,pix;net.route_cache.clear();net._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
   atomic_json(folder/'result.json',result)
 assert tuple(p._version for p in net.parameters())==versions
 status('completed')
if __name__=='__main__':main()
