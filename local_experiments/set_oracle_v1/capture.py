import argparse,json,time,math,hashlib,os,concurrent.futures
from pathlib import Path
import numpy as np
import torch
from ..broad_common import Engine,ROOT as DATA
from ..review_v1.model import Model
from ..review_v1.ops import Summary,SummaryOps,RouteOps,Config,haar
from ..review_v1.ablation import mass_scores
from ..review_v1.runtime import SelectedAttention
from ..rollout_suite import generate_chunk
from ..group_response.core import Runtime
from ..group_response.run import BASE,PRE
from .common import *
def thash(x):return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def arr(x):return x.detach().cpu().numpy()
def save_event(meta,residual,online,temporary,checks):
 eid=meta['event_id'];r=ROOT/'contributions'/eid;r.mkdir(exist_ok=False);np.savez_compressed(r/'residuals_fp64.npz',**residual);np.savez_compressed(r/'summary_and_route_features.npz',**online);np.savez_compressed(ROOT/'_validation_spool'/f'{eid}.npz',**temporary)
 hashes={n:sha(r/n) for n in ('residuals_fp64.npz','summary_and_route_features.npz')};dump(r/'identity.json',dict(meta,files=hashes));dump(r/'capture_validation.json',checks);dump(r/'READY.json',dict(event_id=eid,files=hashes,complete=True));print('SAVED',eid,flush=True)
class Collector:
 def __init__(self,engine,model,case,seed):self.engine=engine;self.model=model;self.case=case;self.seed=seed;self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=2);self.pending=[];self.count=0
 @torch.no_grad()
 def __call__(self,l,q,kh,vh,store,start):
  c=self.engine.current['chunk'];step=self.engine.current['step']
  if c not in (2,5,8) or step not in (0,2) or l not in (1,15,29):return
  h=head(l,c,self.seed,step);hp=store.n;kk=kh[0].transpose(0,1).reshape(12,-1,64,128);vv=vh[0].transpose(0,1).reshape_as(kk)
  cur=SummaryOps.build(self.model[l],kk[:,hp:],vv[:,hp:],cheap=True);hist=store.summary.slice(0,hp);s=Summary(*(torch.cat((getattr(hist,n),getattr(cur,n)),1)[h:h+1] for n in ('keys','mean','detail','stats')));basis=self.model[l].basis[h:h+1];theta=self.model[l].theta[h:h+1];k=kk[h];v=vv[h]
  ids=torch.cat((store.token_ids[h,:hp],torch.arange(start,start+4608,device=q.device).reshape(72,64)),0);assert ids.unique().numel()==ids.numel();assert torch.all(ids[:hp]<start) and torch.all(ids[hp:]>=start)
  for ti in (0,1):
   meta=event(self.case,self.seed,c,l,step,ti);t=meta['tile'];query=q[0,t*64:(t+1)*64,h];P=k.shape[0];own=hp+t;assert P==meta['P'];eid=meta['event_id'];assert not (ROOT/'contributions'/eid).exists()
   qids=torch.arange(start+t*64,start+(t+1)*64,device=q.device);offset=torch.linspace(0,63,16,device=q.device).round().long();f64,online_logz=RouteOps.features(query[None],s);f16,_=RouteOps.features(query[None,:, :].index_select(1,offset),s);risk=(f16*theta[:,None,None]).sum(-1).mean(1)[:,None];mass=mass_scores(query[None].index_select(1,offset),s,1,16);own_tensor=torch.tensor([[own]],device=q.device);masks={n:RouteOps.select(score,own_tensor,Config())['mask'][0,0] for n,score in (('mass',mass),('risk',risk))}
   qs=query.double();kd=k.double();vd=v.double();reused=False;oldpath=Path(f'/tmp/e11_results/contributions/{self.case}_s{self.seed}_c{c}_l{l}_s{step}.npz')
   if ti==0 and oldpath.exists():
    snap=torch.load(Path('/tmp/e11_results/snapshots')/(oldpath.stem+'.pt'),weights_only=True);assert snap['head']==h and snap['tile']==t
    assert torch.equal(query.cpu(),snap['q'][h]) and torch.equal(k.cpu(),snap['k'][h]) and torch.equal(v.cpu(),snap['v'][h])
    for n in ('keys','mean','detail','stats'):assert torch.equal(getattr(s,n)[0].cpu(),snap['summary'][n][h])
    assert torch.equal(basis[0].cpu(),snap['basis'][h]);cached=dict(np.load(oldpath));N,Z,Nh,Zh,scale=(torch.from_numpy(cached[n]).to(q.device) for n in ('N','Z','Nh','Zh','log_scale'));reused=True;source_sha=sha(oldpath)
    for n in masks:assert np.array_equal(arr(masks[n]),cached[n])
    del snap,cached
   else:
    raw=torch.einsum('qd,ptd->qpt',qs,kd)/math.sqrt(128);rep=torch.einsum('qd,prd->qpr',qs,s.keys[0].double())/math.sqrt(128);scale=torch.maximum(raw.flatten(1).amax(1),rep.flatten(1).amax(1)+math.log(16));w=(raw-scale[:,None,None]).exp();rw=(rep+math.log(16)-scale[:,None,None]).exp();Z=w.sum(-1);Zh=rw.sum(-1);N=torch.einsum('qpt,ptd->qpd',w,vd);contrast=torch.einsum('qpr,ar->qpa',rw,haar(q).double());code=torch.einsum('qpa,pak->qpk',contrast,s.detail[0].double());Nh=Zh[:,:,None]*s.mean[0].double()[None]+torch.einsum('qpk,dk->qpd',code,basis[0].bfloat16().double());source_sha=None
   zd=Z.sum(1);od=N.sum(1)/zd[:,None];ze=Z/zd[:,None];zh=Zh/zd[:,None];d=(Z-Zh)/zd[:,None];e=(N-Nh)/zd[:,None,None]-od[:,None,:]*d[:,:,None];energy=od.square().sum()+1e-12
   residual=dict(e=arr(e),d=arr(d),exact_mass=arr(ze),summary_mass=arr(zh),dense_output=arr(od),dense_logz=arr(zd.log()+scale),C=arr(energy),own=np.array(own),K=np.array(meta['K']),mass=arr(masks['mass']),risk=arr(masks['risk']))
   ident=dict(q=thash(query),k=thash(k),v=thash(v),summary={n:thash(getattr(s,n)) for n in ('keys','mean','detail','stats')},basis=thash(basis),theta=thash(theta),token_ids=thash(ids),query_token_ids=thash(qids))
   rawrep=torch.einsum('qd,prd->qpr',query.float(),s.keys[0].float())/math.sqrt(128);mass_q=rawrep.flatten(1).softmax(-1).reshape(64,P,4).sum(-1)
   online=dict(query=arr(query.float()),basis=arr(basis[0].bfloat16().float()),theta=arr(theta[0]),risk_features_q64=arr(f64[0]),risk_features_probe16=arr(f16[0]),risk_scores=arr(risk[0,0]),mass_scores=arr(mass[0,0]),risk_rank=arr(risk[0,0].argsort(descending=True)),mass_rank=arr(mass[0,0].argsort(descending=True)),mass_per_query=arr(mass_q),exact_page_logz=arr(Z.log()+scale[:,None]),summary_page_logz=arr(Zh.log()+scale[:,None]),online_summary_logz=arr(online_logz[0]+math.log(16)),probe_indices=arr(offset),token_ids=arr(ids),query_token_ids=arr(qids),page_is_history=np.arange(P)<hp,page_age=arr(c-ids[:,0]//4608),token_latent_frame=arr(ids//1536),token_y=arr(ids%1536//48),token_x=arr(ids%48))
   for n in ('keys','mean','detail','stats'):online['summary_'+n]=arr(getattr(s,n)[0].float())
   checks=[];rng=np.random.default_rng(17)
   for name,mask in list(masks.items())+[(f'random_budget{budget}',torch.tensor(np.isin(np.arange(P),np.r_[own,rng.choice(np.delete(np.arange(P),own),budget-1,replace=False)]),device=q.device)) for budget in (1,meta['K'],max(1,P//4),P)]:
    mixedz=torch.where(mask[None],Z,Zh).sum(1);mixedn=torch.where(mask[None,:,None],N,Nh).sum(1);out=mixedn/mixedz[:,None];true=((out-od).square().sum()/energy).item();E=e[:,~mask].sum(1);zn=torch.where(mask[None],ze,zh).sum(1);res=(E.square().sum(-1)/zn.square()).sum()/energy;delta=abs(true-res.item());assert delta<max(1e-11,abs(true)*1e-8);assert torch.isfinite(out).all();checks.append(dict(mask=name,direct_fp64_nmse=true,residual_nmse=res.item(),absolute_difference=delta))
   selection=dict(mask=masks['risk'][None,None],ids=torch.where(masks['risk'])[0][None,None],fixed_count=meta['K']);deployed=SelectedAttention.run(query[None],k[None],v[None],s,basis,selection)[0].double();mixed=torch.where(masks['risk'][None,:,None],N,Nh).sum(1)/torch.where(masks['risk'][None],Z,Zh).sum(1)[:,None];fp32=dict(path='original FP32 SelectedAttention; BF16 output',nmse_to_fp64_mixed=float((deployed-mixed).square().sum()/(mixed.square().sum()+1e-12)),nmse_to_fp64_dense=float((deployed-od).square().sum()/energy))
   if self.seed==101 and c==5 and step==0 and l==15 and ti==0:
    torch.save(dict(q=query.cpu(),k=k.cpu(),v=v.cpu(),summary={n:getattr(s,n).cpu() for n in ('keys','mean','detail','stats')},basis=basis.cpu(),ids=ids.cpu()),ROOT/'representatives'/f'{eid}.pt')
   meta.update(own=own,history_pages=hp,hashes=ident,reused_contributions=reused,reused_source_sha256=source_sha,original_precision='BF16',reference_precision='FP64',trajectory='G-KVR L3 self-consistent risk masks, clean Dense',start_token=start)
   temp=dict(N=arr(N),Z=arr(Z),Nh=arr(Nh),Zh=arr(Zh));self.pending.append(self.pool.submit(save_event,meta,residual,online,temp,dict(passed=True,tests=checks,fp32=fp32,token_identity_unique=True,own_query_ids_match=torch.equal(ids[own],qids),reused=reused)))
   self.count+=1
   if len(self.pending)>3:self.pending.pop(0).result()
 def finish(self):
  for f in self.pending:f.result()
  self.pool.shutdown();assert self.count==36
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;r=ROOT/'trajectories'/f'{a.case}_s{a.seed}';r.mkdir(exist_ok=True);dump(r/'status.json',dict(status='loading',pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')));engine=Engine();engine.wrapper.requires_grad_(False);model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False);collector=Collector(engine,model,a.case,a.seed);rt=Runtime(engine,model,'G-KVR','L3',collector);emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda();latents=[];begin=time.monotonic()
 for c in range(9):
  z=generate_chunk(engine,emb,a.seed,c);assert torch.isfinite(z).all();latents.append(z.cpu());dump(r/'status.json',dict(status='generating',chunk=c,events=collector.count,elapsed_seconds=time.monotonic()-begin));print('CHUNK',a.case,a.seed,c,flush=True)
 collector.finish();z=torch.cat(latents,1);torch.save(z,r/'latents.pt');reference=Path(f'/tmp/group_response_results/video/{a.case}_s{a.seed}/G-KVR/L3/latents.pt');equal=None
 if reference.exists():equal=torch.equal(z,torch.load(reference,weights_only=True));assert equal,'Baseline trajectory changed'
 dump(r/'result.json',dict(case=a.case,seed=a.seed,events=collector.count,wall_seconds=time.monotonic()-begin,matching_previous_latents=equal,latent_sha256=thash(z),token_identity=rt.diagnostics()['identity_checks']));rt.close();dump(r/'status.json',dict(status='complete',events=36))
if __name__=='__main__':main()
