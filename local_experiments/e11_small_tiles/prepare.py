import argparse,math,json,hashlib
from pathlib import Path
import numpy as np,torch
from ..review_v1.ops import Summary,SummaryOps,RouteOps,haar,deployed
from ..review_v1.model import Model
from ..group_response.run import BASE,PRE
ROOT=Path('/tmp/e11_small_tiles')
def build(m,k,v,cheap):
 k=deployed(k);v=deployed(v);h,p,t,d=k.shape;g=t//4;assert t in (4,16,64)
 kg=k.reshape(h,p,4,g,d);vg=v.reshape_as(kg)
 if cheap:
  f=torch.einsum('hptd,hrd->hptr',k-k.mean(2,keepdim=True),m.bank)/math.sqrt(d);f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
 else:f=m.group_features(k,v)
 f=f.reshape(h,p,4,g,32)
 z=torch.einsum('hprtd,hrd->hprt',kg-kg.mean(3,keepdim=True),m.pool_k)+torch.einsum('hprtd,hrd->hprt',vg-vg.mean(3,keepdim=True),m.pool_v)+torch.einsum('hprtf,hrf->hprt',f,m.pool_feature)
 a=(2*torch.tanh((z-z.mean(-1,keepdim=True))/2)).softmax(-1);pk=(a[...,None]*kg).sum(3);pv=(a[...,None]*vg).sum(3);pk=deployed(pk-pk.mean(2,keepdim=True)+k.mean(2,keepdim=True));mu=deployed(v.mean(2));basis=deployed(m.basis)
 code=torch.einsum('hprd,hdk->hprk',pv,basis);detail=deployed(torch.einsum('ar,hprk->hpak',haar(k),code)/4);recon=mu[:,:,None]+torch.einsum('ar,hpak,hdk->hprd',haar(k),detail,basis);km=kg.mean(3);vm=vg.mean(3);st=k.new_zeros(h,p,4,12)
 st[...,:8]=(kg-km[:,:,:,None]).square().mean(3).reshape(h,p,4,8,16).mean(-1);st[...,8]=(km-pk).norm(dim=-1);st[...,9]=(vm-recon).norm(dim=-1);st[...,10]=(vg-vm[:,:,:,None]).square().sum(-1).mean(3).sqrt();st[:,:,0,11]=(v.mean(2)-mu).norm(dim=-1);st[:,:,1,11]=v.norm(dim=-1).amax(2)
 return Summary(*(x.bfloat16().contiguous() for x in (pk,mu,detail,st)))
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 path=Path('/tmp/e11_results/snapshots')/f'horse_field_s101_c5_l{a.layer}_s0.pt';snap=torch.load(path,weights_only=True);model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False);m=model[a.layer];head=snap['head'];q=snap['q'].cuda();rawk=snap['k'].cuda();rawv=snap['v'].cuda();hist=snap['history_pages'];origP=rawk.shape[1];budget=int(origP*.1)*64
 base=dict(np.load(Path('/tmp/e11_results/contributions')/(path.stem+'.npz')));od=base['N'].sum(1)/base['Z'].sum(1)[:,None]
 # The generalized implementation must reproduce the original build at Page64.
 checks=[]
 for cheap in (False,True):
  kk=rawk[:,:8];vv=rawv[:,:8];new=build(m,kk,vv,cheap);old=SummaryOps.build(m,kk,vv,cheap=cheap)
  assert all(torch.equal(getattr(new,n),getattr(old,n)) for n in ('keys','mean','detail','stats'))
 for size in (16,4):
  root=ROOT/f'l{a.layer}'/f'q{size}_k{size}';root.mkdir(parents=True,exist_ok=True);k=rawk.reshape(12,-1,size,128);v=rawv.reshape_as(k);nh=hist*64//size;P=k.shape[1];K=budget//size
  assert torch.equal(k.flatten(1,2),rawk.flatten(1,2)) and torch.equal(v.flatten(1,2),rawv.flatten(1,2))
  hs=build(m,k[:,:nh],v[:,:nh],False);cs=build(m,k[:,nh:],v[:,nh:],True);s=Summary(*(torch.cat((getattr(hs,n),getattr(cs,n)),1) for n in ('keys','mean','detail','stats')))
  torch.save(dict(summary={n:getattr(s,n)[head].cpu() for n in ('keys','mean','detail','stats')},basis=m.basis[head].detach().cpu(),source_snapshot=str(path),head=head),root/'summary.pt')
  qq=q[head].double();raw=torch.einsum('qd,ptd->qpt',qq,k[head].double())/math.sqrt(128);rep=torch.einsum('qd,prd->qpr',qq,s.keys[head].double())/math.sqrt(128);g=size/4
  scale=torch.maximum(raw.flatten(1).amax(1),rep.flatten(1).amax(1)+math.log(g));rw=(raw-scale[:,None,None]).exp();sw=(rep+math.log(g)-scale[:,None,None]).exp();Z=rw.sum(-1);Zh=sw.sum(-1);N=torch.einsum('qpt,ptd->qpd',rw,v[head].double());contrast=torch.einsum('qpr,ar->qpa',sw,haar(q).double());code=torch.einsum('qpa,pak->qpk',contrast,s.detail[head].double());Nh=Zh[:,:,None]*s.mean[head].double()[None]+torch.einsum('qpk,dk->qpd',code,m.basis[head].bfloat16().double())
  data={n:x.cpu().numpy() for n,x in dict(N=N,Z=Z,Nh=Nh,Zh=Zh,log_scale=scale).items()};dense=data['N'].sum(1)/data['Z'].sum(1)[:,None];err=float(np.max(np.abs(dense-od)));assert err<1e-10
  tiles=64//size;offsets=torch.linspace(0,size-1,min(16,size),device=q.device).round().long();probes=q.reshape(12,tiles,size,128).index_select(2,offsets).flatten(1,2);f,_=RouteOps.features(probes,s);risk=(f*m.theta[:,None,None]).sum(-1).reshape(12,tiles,len(offsets),P).mean(2)
  logits=torch.einsum('hqd,hprd->hqpr',probes.float(),s.keys.float())/math.sqrt(128);mass=logits.flatten(2).softmax(-1).reshape(12,tiles,len(offsets),P,4).sum(-1).mean(2)
  for ti in range(tiles):
   own=snap['own']*64//size+ti;small={n:x[ti*size:(ti+1)*size] for n,x in data.items()}
   for name,score in [('mass',mass),('risk',risk)]:
    score=score[head,ti].clone();score[own]=torch.inf;idx=score.topk(K).indices;mask=np.zeros(P,bool);mask[idx.cpu().numpy()]=True;small[name]=mask
   small.update(own=np.array(own),K=np.array(K),page_size=np.array(size),query_offset=np.array(ti*size));np.savez_compressed(root/f't{ti}.npz',**small)
  meta=dict(layer=a.layer,head=head,case='horse_field',seed=101,chunk=5,step=0,Q=size,K_page_tokens=size,P=P,selected_pages=K,exact_tokens_per_query=K*size,total_tokens=P*size,nominal_density=.1,actual_density=K/P,query_tiles=tiles,evaluated_queries=64,representatives=4,representative_mass=g,rank=8,online_probes=min(16,size),dense_identity_max_abs=err,token_identity_unchanged=True,page64_build_bitwise_identical=True,source_snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((root/'summary.pt').read_bytes()).hexdigest(),summary_note='Frozen pooling formula generalized to four subgroups of page_size/4. New BF16 summaries; no retraining or regrouping. R4 increases total representative count as pages shrink.')
  (root/'metadata.json').write_text(json.dumps(meta,indent=2));print('PREPARED',meta,flush=True)
 (ROOT/f'l{a.layer}'/'prepared.json').write_text(json.dumps(dict(status='complete',layer=a.layer)))
if __name__=='__main__':main()
