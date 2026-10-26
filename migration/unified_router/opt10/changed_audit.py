"""Audit every changed query in selected real trajectory blocks, without fixed-query sampling."""
import argparse,os,json,gc
import torch
from ..common import config,atomic_json,update,paired_seed
from ..opt6_guarded import load_guarded_head
from ..opt9.network import PackedRoute
from .network import configure
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,_=config(a.directory);out=root/'optimization10/changed_audit';out.mkdir(parents=True,exist_ok=True)
 rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));paired_seed(0)
 def status(stage,**kw):update(out/f'rank{rank}.json',status=stage,**kw);print(stage,kw,flush=True)
 status('loading');net,_=load_guarded_head(root);base_prepare=net.prepare;ref=PackedRoute(batch=512)
 from streaming_sparse.v08.replay import ReplayDataset
 from streaming_sparse.v08.rollout import trajectory
 from streaming_sparse.capture.capture_runner import sdpa_backend
 ds=ReplayDataset('val',require_complete=True);pids=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids'];rows=[]
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def observe(layer,start,step,q,k,v,sketch,prepared,mask,y):
  if layer not in [1,24,29]or step!=3:return
  state=ref.begin(prepared,sketch);st=q.new_tensor(float(step),dtype=torch.float32)
  rm=torch.cat([ref.run(layer,q[:,i:i+512],sketch,state,st,net.router)for i in range(0,q.shape[1],512)],1)
  changed=(mask!=rm).any(-1);sse=energy=0.;max_nmse=0.;checked=0
  for h in range(q.shape[0]):
   ids=changed[h].nonzero().flatten()
   for offset in range(0,len(ids),128):
    ii=ids[offset:offset+128];qq=q[h,ii][None,None];kk=k[h][None,None];vv=v[h][None,None]
    outs=[]
    for mm in [rm,mask]:
     visible=mm[h,ii].repeat_interleave(sketch.costs.long(),-1)[None,None]
     outs.append(torch.nn.functional.scaled_dot_product_attention(qq,kk,vv,attn_mask=visible)[0,0].float())
    num=(outs[1]-outs[0]).square().sum(-1);den=outs[0].square().sum(-1);sse+=float(num.sum());energy+=float(den.sum());max_nmse=max(max_nmse,float((num/den.clamp_min(1e-20)).max()));checked+=len(ii)
  assert checked==int(changed.sum())
  rows.append(dict(layer=layer,start=start,step=step,total_queries=mask.shape[0]*mask.shape[1],changed_queries=checked,sse=sse,reference_energy=energy,changed_query_aggregate_nmse=sse/energy if energy else 0.,worst_changed_query_nmse=max_nmse))
 net.inspect_callback=observe
 with sdpa_backend():
  for pid in pids[rank::world]:
   sample=ds.load(next(i for i,r in enumerate(ds.rows)if r['prompt_id']==pid and r['generation_seed']==0));embeds=sample['prompt_embeds'].cuda().to(torch.bfloat16)
   for variant in ['Head','Moments','Compact']:
    f=out/f'{pid}.{variant}.json'
    if f.exists():continue
    status('auditing',prompt=pid,variant=variant);configure(net,variant,base_prepare);net.set_mode('K32-Set');rows.clear();chunks,_=trajectory(net,embeds,pid,0,True);latent=torch.cat(chunks,1)
    expected=torch.load(root/f'optimization10/quality/{pid}.seed0/{variant}.latent.pt',map_location='cpu',weights_only=True);assert torch.equal(latent,expected)
    atomic_json(f,dict(prompt_id=pid,variant=variant,latent_matches_quality=True,audit=list(rows)));del chunks,latent,expected;net._eval_sketch.clear();net.route_cache.clear();gc.collect()
 status('completed')
if __name__=='__main__':main()
