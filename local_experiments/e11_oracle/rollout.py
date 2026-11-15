"""Matched sampled intervention. Each method evolves its own clean cache and masks."""
import argparse,math,json,time,os,gc,subprocess,concurrent.futures
from pathlib import Path
import torch,numpy as np
from ..group_response.core import Runtime
from ..group_response.run import BASE,PRE
from ..broad_common import Engine,ROOT as DATA
from ..review_v1.model import Model
from ..review_v1.ops import Summary,SummaryOps,RouteOps,Config,haar
from ..review_v1.ablation import mass_scores
from ..review_v1.runtime import SelectedAttention
from ..rollout_suite import generate_chunk
from ..stream_eval_common import native_vae,decode,save_video
from ..high_sparse_experiment import prefix_quality
from ..learned_cache.io import atomic
from .contributions import diagnostic
ROOT=Path('/tmp/e11_results')

def one_contribution(q,k,v,s,basis):
 raw=torch.einsum('qd,ptd->qpt',q.double(),k.double())/math.sqrt(128);rep=torch.einsum('qd,prd->qpr',q.double(),s.keys[0].double())/math.sqrt(128)
 scale=torch.maximum(raw.flatten(1).amax(1),rep.flatten(1).amax(1)+math.log(16));weights=(raw-scale[:,None,None]).exp();rw=(rep+math.log(16)-scale[:,None,None]).exp()
 Z=weights.sum(-1);Zh=rw.sum(-1);N=torch.einsum('qpt,ptd->qpd',weights,v.double());code=torch.einsum('qpa,pak->qpk',torch.einsum('qpr,ar->qpa',rw,haar(q).double()),s.detail[0].double());Nh=Zh[:,:,None]*s.mean[0].double()[None]+torch.einsum('qpk,dk->qpd',code,basis[0].bfloat16().double())
 return {n:z.cpu().numpy() for n,z in dict(N=N,Z=Z,Nh=Nh,Zh=Zh,log_scale=scale).items()}

class Sampled(Runtime):
 def __init__(self,engine,model,case,method,layers,root,limit):
  self.case=case;self.method=method;self.layers=set(layers);self.root=root;self.limit=limit;self.interventions=[];self.original_dense=engine.dense
  super().__init__(engine,model,'G-KVR','Dense');engine.dense=self.attend
 def forward(self,l,*args):
  self.active_layer=l
  try:return super().forward(l,*args)
  finally:self.active_layer=None
 def attend(self,q,kh,vh,*args,**kwargs):
  dense=self.original_dense(q,kh,vh,*args,**kwargs);l=getattr(self,'active_layer',None);c=self.engine.current['chunk'];step=self.engine.current['step']
  if l not in self.layers or c not in (2,5,8) or step not in (0,2):return dense
  store=self.stores[l];p=store.n;m=self.model[l];k=kh[0].transpose(0,1).reshape(12,-1,64,128);v=vh[0].transpose(0,1).reshape_as(k);query=q[0].transpose(0,1)
  cur=SummaryOps.build(m,k[:,p:],v[:,p:],cheap=True);history=store.summary.slice(0,p);s=Summary(*(torch.cat((getattr(history,n),getattr(cur,n)),1) for n in ('keys','mean','detail','stats')))
  t0=(17*l+13*c+7*step+101)%72;tiles=[t0,(t0+36)%72];qs=torch.stack([query[:,t*64:(t+1)*64] for t in tiles],1);probes=qs[:,:,torch.linspace(0,63,16,device=q.device).round().long()].flatten(1,2);f,_=RouteOps.features(probes,s)
  scores={'O-RISK':(f*m.theta[:,None,None]).sum(-1).reshape(12,2,16,p+72).mean(2),'O-MASS':mass_scores(probes,s,2,16)};own=(torch.tensor(tiles,device=q.device)+p)[None].expand(12,-1);selections={n:RouteOps.select(z,own,Config()) for n,z in scores.items()};K=math.floor((p+72)*.1)
  event=self.root/'events'/f'c{c}_l{l}_s{step}';event.mkdir(parents=True,exist_ok=True)
  atomic(event/'input.pt',dict(q=qs.cpu(),k=k.cpu(),v=v.cpu(),summary={n:getattr(s,n).cpu() for n in ('keys','mean','detail','stats')},basis=m.basis.detach().cpu(),ids=store.token_ids[:,:p].cpu(),tiles=tiles),True)
  pending=[]
  for ti,tile in enumerate(tiles):
   for h in (0,5,11):
    ss=Summary(*(getattr(s,n)[h:h+1] for n in ('keys','mean','detail','stats')));data=one_contribution(qs[h,ti],k[h],v[h],ss,m.basis[h:h+1]);data.update(own=np.array(p+tile),head=np.array(h),mass=selections['O-MASS']['mask'][h,ti].cpu().numpy(),risk=selections['O-RISK']['mask'][h,ti].cpu().numpy());dest=event/f't{tile}_h{h}';dest.mkdir(exist_ok=True);np.savez_compressed(dest/'contributions.npz',**data);pending.append((ti,tile,h,ss,data,dest))
  if self.method=='O-OPT':
   def solve_item(item):
    dest=item[-1];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (dest/'worker.log').open('w') as f:r=subprocess.run([os.sys.executable,'-m','local_experiments.e11_oracle.solve_file',str(dest/'contributions.npz'),'--limit',str(self.limit)],env=env,stdout=f,stderr=subprocess.STDOUT)
    assert r.returncode==0,(dest,r.returncode)
   with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(solve_item,pending))
  output=dense.clone()
  for ti,tile,h,ss,data,dest in pending:
   if self.method=='O-OPT':
    result=json.loads((dest/'solve/result.json').read_text());mask=np.zeros(p+72,bool);mask[result['mask']]=True;status=result['status']
   else:mask=data['mass' if self.method=='O-MASS' else 'risk'];status='ONLINE_ROUTER'
   assert mask.sum()==K and mask[p+tile];ids=np.where(mask)[0];selection=dict(mask=torch.tensor(mask,device=q.device)[None,None],ids=torch.tensor(ids,device=q.device)[None,None],fixed_count=K)
   value=SelectedAttention.run(qs[h:h+1,ti],k[h:h+1],v[h:h+1],ss,m.basis[h:h+1],selection)
   output[0,tile*64:(tile+1)*64,h]=value[0]
   arrays,metrics=diagnostic(data,mask);np.savez_compressed(dest/'D02.npz',mask=mask,**arrays);metrics.update(status=status,head=h,tile=tile,layer=l,chunk=c,step=step,count=K,own=p+tile,method=self.method,selected_pages=ids.tolist());atomic(dest/'local.json',metrics);self.interventions.append(metrics)
  # Explicitly ensure no additional head/query received a sparse intervention.
  active=torch.zeros((4608,12),device=q.device,dtype=torch.bool)
  for tile in tiles:
   for h in (0,5,11):active[tile*64:(tile+1)*64,h]=True
  assert torch.equal(output[0][~active],dense[0][~active])
  return output
 def close(self):self.engine.dense=self.original_dense;super().close()

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--method',choices=['O-MASS','O-RISK','O-OPT'],required=True);p.add_argument('--layer',type=int,default=0);p.add_argument('--limit',type=float,required=True);a=p.parse_args()
 layers=[a.layer] if a.layer else [1,8,15,22,29];level=f'L2_l{a.layer}' if a.layer else 'L3_sampled';root=ROOT/'video'/a.case/a.method/level;root.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 engine=Engine();engine.wrapper.requires_grad_(False);model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False);rt=Sampled(engine,model,a.case,a.method,layers,root,a.limit);emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda();zs=[];start=time.monotonic()
 for c in range(9):
  atomic(root/'status.json',dict(status='generating',chunk=c));zs.append(generate_chunk(engine,emb,101,c).cpu());print(a.case,a.method,level,c,flush=True)
 z=torch.cat(zs,1);atomic(root/'latents.pt',z,True);interventions=rt.interventions;assert len(interventions)==36*len(layers);atomic(root/'interventions.json',interventions);rt.close();del rt,engine,model;gc.collect();torch.cuda.empty_cache()
 refpath=Path('/tmp/group_response_results/video')/f'{a.case}_s101/G-KVR/Dense/latents.pt';ref=torch.load(refpath,weights_only=True);latent=float((z.float()-ref.float()).square().sum()/ref.float().square().sum());latent_per_frame=((z.float()-ref.float()).square().sum(tuple(i for i in range(z.ndim) if i!=1))/ref.float().square().sum(tuple(i for i in range(z.ndim) if i!=1))).tolist()
 vae=native_vae();frames=decode(vae,z.cuda(),105);reference=decode(vae,ref.cuda(),105);save_video(root/'video.mp4',frames);os.environ['TORCH_HOME']=str(Path('artifacts/rollout_acceptance_20261030/torch_cache').resolve());import lpips
 metric=lpips.LPIPS(net='alex').eval().requires_grad_(False);quality=prefix_quality(frames,reference,metric);quality['per_frame']['psnr']=[100. if x==0 else -10*math.log10(x) for x in quality['per_frame']['mse']]
 proof=sum(x['status']=='GLOBAL_TOL' for x in interventions);atomic(root/'result.json',dict(status='complete',method=a.method,case=a.case,seed=101,level=level,layers=layers,chunks=9,frames=105,latent_nmse=latent,latent_per_frame_nmse=latent_per_frame,quality=quality,coverage='sampled identical chunks2/5/8 steps0/2 heads0/5/11 tiles t0/t1; all other head-tiles Dense',instances=len(interventions),fraction_of_full_sparse_headtiles=len(interventions)/(24192*len(layers)),proof_count=proof,proof_fraction=proof/len(interventions) if a.method=='O-OPT' else None,run_label=('global_tol_rollout' if proof==len(interventions) else 'bounded_search_rollout') if a.method=='O-OPT' else 'sampled_online_route',reference=str(refpath),solver_seconds_per_instance=a.limit,wall_seconds=time.monotonic()-start));atomic(root/'status.json',dict(status='complete'))
if __name__=='__main__':main()
