"""FP64 contributions from deployed BF16 input and actual stored summaries."""
import math,argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from ..review_v1.ops import Summary,RouteOps,Config,haar
from ..review_v1.ablation import mass_scores
from ..review_v1.runtime import SelectedAttention
ROOT=Path('/tmp/e11_results')
@torch.no_grad()
def extract(path,device='cuda'):
 snap=torch.load(path,weights_only=True);h=snap['head'];q=snap['q'].to(device);k=snap['k'].to(device);v=snap['v'].to(device);s=Summary(*(snap['summary'][n].to(device) for n in ('keys','mean','detail','stats')));basis=snap['basis'].to(device)
 p=k.shape[1];own=torch.full((12,1),snap['own'],device=device,dtype=torch.long);probes=q[:,torch.linspace(0,63,16,device=device).round().long()]
 f,_=RouteOps.features(probes,s);risk=(f*snap['theta'].to(device)[:,None,None]).sum(-1).mean(1)[:,None];mass=mass_scores(probes,s,1,16)
 masks={name:RouteOps.select(score,own,Config())['mask'][h,0].cpu().numpy() for name,score in [('mass',mass),('risk',risk)]}
 # All heads are retained for a correctly normalized, single-head intervention post-W_O metric.
 od_all=[]
 for j in range(12):
  scores=q[j].double()@k[j].flatten(0,1).double().T/math.sqrt(128);od_all.append(scores.softmax(-1)@v[j].flatten(0,1).double())
 od_all=torch.stack(od_all);query=q[h].double();kk=k[h].double();vv=v[h].double()
 raw=torch.einsum('qd,ptd->qpt',query,kk)/math.sqrt(128);rep=torch.einsum('qd,prd->qpr',query,s.keys[h].double())/math.sqrt(128)
 # ONE scale per query, shared across every exact and representative contribution.
 scale=torch.maximum(raw.flatten(1).amax(1),rep.flatten(1).amax(1)+math.log(16))
 weights=(raw-scale[:,None,None]).exp();rw=(rep+math.log(16)-scale[:,None,None]).exp()
 Z=weights.sum(-1);Zh=rw.sum(-1);N=torch.einsum('qpt,ptd->qpd',weights,vv)
 contrasts=torch.einsum('qpr,ar->qpa',rw,haar(query).double());code=torch.einsum('qpa,pak->qpk',contrasts,s.detail[h].double())
 Nh=Zh[:,:,None]*s.mean[h].double()[None]+torch.einsum('qpk,dk->qpd',code,basis[h].bfloat16().double())
 data={n:z.cpu().numpy() for n,z in dict(N=N,Z=Z,Nh=Nh,Zh=Zh,log_scale=scale,dense_heads=od_all,output_weight=snap['output_weight'].double(),output_bias=torch.zeros(1536,dtype=torch.double) if snap['output_bias'] is None else snap['output_bias'].double()).items()}
 data.update(mass=masks['mass'],risk=masks['risk'],own=np.array(snap['own']),head=np.array(h));assert np.isfinite(N.cpu().numpy()).all() and bool((Z>0).all()) and bool((Zh>0).all())
 root=ROOT/'contributions';root.mkdir(exist_ok=True,parents=True);np.savez_compressed(root/(path.stem+'.npz'),**data)
 meta={n:snap[n] for n in ('case','seed','chunk','layer','step','history_pages','head','tile','own','summary_source')};meta.update(snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_log_range=float(raw.max()-raw.min()),summary_log_range=float(rep.max()-rep.min()),common_scale_per_query=True)
 (root/(path.stem+'.json')).write_text(json.dumps(meta,indent=2));return data

def diagnostic(data,mask):
 N,Z,Nh,Zh=(data[n] for n in ('N','Z','Nh','Zh'));od=N.sum(1)/Z.sum(1)[:,None];e=(N-Nh)-od[:,None,:]*(Z-Zh)[:,:,None];tail=e[:,~mask];E=tail.sum(1)
 A=np.square(tail).sum((1,2));norm=np.square(E).sum(1);zm=np.where(mask[None,:],Z,Zh).sum(1);zd=Z.sum(1);out=np.where(mask[None,:,None],N,Nh).sum(1)/zm[:,None];direct=out-od
 identity=float(np.max(np.abs(direct+E/zm[:,None])));assert identity<1e-8*max(1.,float(np.max(np.abs(direct)))),identity
 rng=np.random.default_rng(17);pairs=rng.integers(0,tail.shape[1],(256,2));ea=tail[:,pairs[:,0]];eb=tail[:,pairs[:,1]];cos=(ea*eb).sum(-1)/np.maximum(np.linalg.norm(ea,axis=-1)*np.linalg.norm(eb,axis=-1),np.finfo(float).tiny)
 result=dict(A_over_mixed_Z2=A/zm**2,C_over_mixed_Z2=(norm-A)/zm**2,A_over_dense_Z2=A/zd**2,C_over_dense_Z2=(norm-A)/zd**2,A=A,C=norm-A,kappa=norm/(A+1e-12),Jq=norm/zm**2,fixed_dense_denominator_Jq=norm/zd**2,denominator_ratio=zm/zd,log_denominator_delta=np.log(zm)-np.log(zd),pair_cosines=cos,pair_indices=pairs,output=out)
 energy=np.square(od).sum()+1e-12;metrics=dict(nmse=float(np.square(direct).sum()/energy),identity_max_abs=identity)
 if 'dense_heads' in data:
  h=int(data['head']);W=data['output_weight'];densepost=data['dense_heads'].transpose(1,0,2).reshape(64,1536)@W.T+data['output_bias'];error=direct@W[:,h*128:(h+1)*128].T
  metrics['post_WO_single_head_nmse']=float(np.square(error).sum()/(np.square(densepost).sum()+1e-12))
 return result,metrics

@torch.no_grad()
def selected_check(snapshot,mask,data):
 snap=torch.load(snapshot,weights_only=True);h=snap['head'];q=snap['q'][h:h+1].cuda();k=snap['k'][h:h+1].cuda();v=snap['v'][h:h+1].cuda();s=Summary(*(snap['summary'][n][h:h+1].cuda() for n in ('keys','mean','detail','stats')));basis=snap['basis'][h:h+1].cuda();ids=np.where(mask)[0]
 selection=dict(mask=torch.tensor(mask,device='cuda')[None,None],ids=torch.tensor(ids,device='cuda')[None,None],fixed_count=len(ids))
 out=SelectedAttention.run(q,k,v,s,basis,selection).double().cpu().numpy()[0];details,metrics=diagnostic(data,mask);ref=details['output'];od=data['N'].sum(1)/data['Z'].sum(1)[:,None]
 return dict(selected_path='original FP32 SelectedAttention with deployed BF16 output',selected_vs_fp64_nmse=float(np.square(out-ref).sum()/(np.square(ref).sum()+1e-12)),selected_vs_fp64_max_abs=float(np.abs(out-ref).max()),selected_vs_fp64_dense_nmse=float(np.square(out-od).sum()/(np.square(od).sum()+1e-12)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');a=p.parse_args();torch.set_num_threads(4)
 for path in a.paths:extract(Path(path));print('EXTRACTED',path,flush=True)
