"""Paired native-frame validation against Opt7-Full; actual-QKV audits."""
import argparse,gc,json,os
import torch
from ..common import config,atomic_json,update,paired_seed,sha
from ..opt6_guarded import load_guarded_head,GuardedHeadRoute
from .guarded import configure_guarded as configure
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,m=config(a.directory);out=root/'optimization8/guard_quality';out.mkdir(exist_ok=True)
 rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));paired_seed(0)
 manifest=json.loads((root/'e2e_manifest.json').read_text());seal=json.loads((root/'optimization7/manifest.json').read_text())
 for path,digest in {**seal['code'],**seal['prior_unchanged']}.items():assert sha(path)==digest,path
 def status(stage,**kw):update(out/f'rank{rank}.json',status=stage,rank=rank,**kw);print(stage,kw,flush=True)
 status('loading');net,_=load_guarded_head(root);base_prepare=net.prepare
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 from migration.frozen_quality.worker import pixel_metrics,preview
 from utils.wan_wrapper import WanVAEWrapper
 import lpips
 vae=WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
 metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,pnet_rand=False,eval_mode=True).cuda().eval().requires_grad_(False)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ds=ReplayDataset('val',require_complete=True);assert ds.fingerprint()==manifest['val_fingerprint'];versions=tuple(p._version for p in net.parameters());refroute=GuardedHeadRoute(batch=512)
 def decode(latent):
  pix=vae.decode_to_pixel(latent.cuda().to(torch.bfloat16),use_cache=False)[0].cpu();vae.model.clear_cache();return pix
 audit=[]
 def observe(layer,start,step,q,k,v,sketch,prepared,mask,y):
  if layer not in net.layers:return
  ids=torch.tensor([0,1,7,31,32,51,52,1559,1560,3119,3120,4679],device=q.device)
  visible=mask[:,ids].repeat_interleave(sketch.costs.long(),-1)
  truth=torch.nn.functional.scaled_dot_product_attention(q[:,ids][None],k[None],v[None],attn_mask=visible[None])[0]
  dense=torch.nn.functional.scaled_dot_product_attention(q[:,ids][None],k[None],v[None])[0]
  yy=y[:,ids];assert torch.isfinite(y).all();nmse=float((yy.float()-truth.float()).square().sum()/truth.float().square().sum())
  assert nmse<.01,(layer,start,nmse)
  row=dict(layer=layer,start=start,queries=ids.numel()*12,masked_attention_nmse=nmse,sparse_to_dense_sse=float((yy.float()-dense.float()).square().sum()),dense_energy=float(dense.float().square().sum()))
  if prepared is not None and layer in [1,24,29]and step==3:
   state=refroute.begin(prepared,sketch);st=q.new_tensor(float(step),dtype=torch.float32)
   rm=torch.cat([refroute.run(layer,q[:,i:i+512],sketch,state,st,net.router)for i in range(0,q.shape[1],512)],1)
   row['route_checked_queries']=mask.shape[0]*mask.shape[1];row['route_changed_queries']=int((rm!=mask).any(-1).sum())
   if net.opt8_variant=='Full':assert row['route_changed_queries']==0,row
  audit.append(row)
 net.inspect_callback=observe
 with sdpa_backend():
  for pid in manifest['prompt_ids'][rank::world]:
   folder=out/f'{pid}.seed0';folder.mkdir(exist_ok=True)
   sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
   dl=torch.load(root/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)['Dense'];dense=decode(dl)
   pl=torch.load(root/f'optimization7/quality/{pid}.seed0/GuardCachedSplitCounters.latent.pt',map_location='cpu',weights_only=True);prior=decode(pl)
   result=dict(prompt_id=pid,test_used=False,baseline_vs_dense=pixel_metrics(dense,prior,metric),variants={})
   for variant in ['QKProjGuard','QKFastGuard']:
    status('evaluating',prompt=pid,variant=variant);gc.collect();torch.cuda.empty_cache();configure(net,variant,base_prepare);net.opt8_variant=variant;net.set_mode('U-Sol'if 'Fast' in variant else 'K32-Set');audit.clear()
    chunks,_=trajectory(net,embeds,pid,0,True);latent=torch.cat(chunks,1);same=torch.equal(latent,pl)
    if variant=='Full':assert same,'Exact mean fusion changed the end-to-end latent'
    assert torch.equal(latent[:,:3],dl[:,:3]);pix=prior if same else decode(latent);assert torch.equal(pix[:9],dense[:9])
    records=net.routing_records;assert len(records)==348
    retained=torch.stack([r['retained_H']+r['retained_C']for r in records]).sum().item();denom=sum(r['queries']*(r['history_tokens']+r['current_tokens'])for r in records)
    row=dict(variant=variant,weighted_keep=retained/denom,quality_vs_dense=pixel_metrics(dense,pix,metric),quality_vs_opt7=pixel_metrics(prior,pix,metric),latent_bit_identical=same,first_chunk_identical=True,attention_audit=list(audit))
    torch.save(latent,folder/(variant+'.latent.pt'));atomic_json(folder/(variant+'.json'),row);result['variants'][variant]=row
    if variant!='Full':preview(folder/f'Dense_Opt7_{variant}.mp4',{'Dense':dense,'Opt7-Full':prior,'Opt8':pix},['Dense','Opt7-Full','Opt8'])
    print(variant,'LPIPS',row['quality_vs_dense']['sparse_frames_only']['lpips'],'keep',row['weighted_keep'],flush=True)
    del chunks,latent,pix;net.route_cache.clear();net._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
   atomic_json(folder/'result.json',result)
 assert tuple(p._version for p in net.parameters())==versions
 status('completed')
if __name__=='__main__':main()
