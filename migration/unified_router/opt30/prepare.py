"""Train-only behavior distillation into the unchanged depth-six runtime tree.

Q targets: true dense attention mass over reference K-tree buckets.
K targets: attention received from reference Q-tree buckets, normalized per K.
The reference bins are fixed across calibration events, so signatures align.
No Val/Test samples, gradients through sorting, or runtime dense computation.
"""
import argparse,json,math,time,random
from pathlib import Path
import torch
from .setup import ROOT
from .data import Events
from .model import QueryRouter
from ..opt16.grouping import Tree,labels,order,gather
from ..common import atomic_json,save

@torch.no_grad()
def fit(x,signature,depth=6,prior=None):
 h,n,d=x.shape;nodes=2**depth-1
 weights=x.new_zeros(h,nodes,d,dtype=torch.float32);thresholds=x.new_zeros(h,nodes,dtype=torch.float32);records=[]
 # Root features normalized once; fold normalization into raw-Q/K planes.
 for head in range(h):
  xx=x[head].float();target=signature[head].float();pending=[(0,torch.arange(n,device=x.device))]
  for level in range(depth):
   following=[]
   for node,ids in pending:
    if len(ids)<4:
     following.extend([(2*node+1,ids),(2*node+2,ids[:0])]);continue
    if prior is not None and level<3:
     weights[head,node]=prior.weight[head,node];thresholds[head,node]=prior.threshold[head,node]
     actual=xx[ids]@weights[head,node]>thresholds[head,node]
     following.extend([(2*node+1,ids[~actual]),(2*node+2,ids[actual])]);continue
    yy=target[ids];z=yy-yy.mean(0);cov=z.T@z;direction=torch.zeros(z.shape[1],device=x.device);direction[cov.diag().argmax()]=1
    for _ in range(12):direction=cov@direction;direction/=direction.norm().clamp_min(1e-12)
    score=z@direction
    # Balanced target-space two-means; balancing avoids unusably tiny buckets.
    for _ in range(8):
     right=score>score.median()
     if not right.any()or right.all():break
     center=yy[right].mean(0)-yy[~right].mean(0);score=yy@center
    right=score>score.median()
    if prior is not None:
     previous=xx[ids]@prior.weight[head,node]>prior.threshold[head,node]
     if (right==previous).float().mean()<.5:right=~right
    y=right.float()*2-1
    features=xx[ids];mean=features.mean(0);std=features.std(0,unbiased=False).clamp_min(.05);features=(features-mean)/std
    reg=.05*len(ids);eye=torch.eye(d,device=x.device)
    w=torch.linalg.solve(features.T@features+reg*eye,features.T@y)
    # Refine the separating plane with supervised logistic loss; no optimizer state at runtime.
    b=torch.zeros((),device=x.device)
    for _ in range(24):
     p=(features@w+b).sigmoid();error=p-right.float()
     w-=.1*(features.T@error/len(ids)+.01*w);b-=.1*error.mean()
    plane=w/std;response=xx[ids]@plane;threshold=response.median();actual=response>threshold
    weights[head,node]=plane;thresholds[head,node]=threshold
    records.append(dict(head=head,level=level,node=node,count=len(ids),classification_accuracy=float((actual==right).float().mean())))
    following.extend([(2*node+1,ids[~actual]),(2*node+2,ids[actual])])
   pending=following
 return Tree(weights,thresholds,depth),records

@torch.no_grad()
def blend(old,new,alpha):
 if alpha==0:return old
 norm0=old.weight.norm(dim=-1).clamp_min(1e-12);norm1=new.weight.norm(dim=-1).clamp_min(1e-12)
 w=(1-alpha)*old.weight/norm0[...,None]+alpha*new.weight/norm1[...,None]
 b=(1-alpha)*old.threshold/norm0+alpha*new.threshold/norm1
 return Tree(w.contiguous(),b.contiguous(),old.depth)

@torch.no_grad()
def choose_native(model,events,items,qproposal,kproposal):
 oldq=model.tree('q');oldk=model.tree('k')
 oldq=Tree(oldq.weight.clone(),oldq.threshold.clone(),oldq.depth);oldk=Tree(oldk.weight.clone(),oldk.threshold.clone(),oldk.depth)
 alphas=[0.,.25,.5,1.];qs={a:blend(oldq,qproposal,a)for a in alphas};ks={a:blend(oldk,kproposal,a)for a in alphas};rows=[]
 for qa in alphas:
  for ka in alphas:
   model.set_trees(qs[qa],ks[ka]);parts=[]
   for item in items:
    q,raw,truth,targets=events.gpu(item);out,_=model.deployed(q,raw,targets)
    parts.append((out-truth).square().sum((1,2))/truth.square().sum().clamp_min(1e-20))
   contrib=torch.stack(parts).mean(0);rows.append(dict(q_alpha=qa,k_alpha=ka,nmse=float(contrib.sum()),head_contributions=contrib.tolist()))
   print('native_tree_probe',rows[-1],flush=True)
 # Heads are independent in deployed attention. Choose each head's update instead
 # of letting a harmful head veto every useful head's learned reordering.
 def select(pool):
  picks=[min(pool,key=lambda x:x['head_contributions'][h])for h in range(12)]
  qt=Tree(torch.stack([qs[x['q_alpha']].weight[h]for h,x in enumerate(picks)]),torch.stack([qs[x['q_alpha']].threshold[h]for h,x in enumerate(picks)]),oldq.depth)
  kt=Tree(torch.stack([ks[x['k_alpha']].weight[h]for h,x in enumerate(picks)]),torch.stack([ks[x['k_alpha']].threshold[h]for h,x in enumerate(picks)]),oldk.depth)
  return qt,kt,dict(q_alpha=[x['q_alpha']for x in picks],k_alpha=[x['k_alpha']for x in picks],nmse=sum(x['head_contributions'][h]for h,x in enumerate(picks)))
 qbest,_,qr=select([x for x in rows if x['k_alpha']==0]);_,kbest,kr=select([x for x in rows if x['q_alpha']==0]);jointq,jointk,jr=select(rows)
 model.set_trees(jointq,jointk)
 from .train import evaluate
 checked=evaluate(model,events,items)['nmse'];assert abs(checked-jr['nmse'])<1e-6,(checked,jr)
 model.set_trees(oldq,oldk)
 return qbest,kbest,jointq,jointk,dict(rows=rows,q_only=qr,k_only=kr,joint=jr,joint_verified_nmse=checked,probe_sample_ids=[x['row']['sample_id']for x in items],split='train',selection='Per-head native attention NMSE, disjoint from signature calibration; assembled joint output verified')

@torch.no_grad()
def signature_dispersion(x,sig,tree,tile):
 ix=order(x,tree);z=gather(sig,ix);h,n,d=z.shape;n=n//tile*tile;z=z[:,:n].reshape(h,-1,tile,d)
 return float((z-z.mean(2,keepdim=True)).square().sum(-1).mean())

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',required=True,type=int);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 folder=ROOT/f'grouping/layer{a.layer:02d}';folder.mkdir(parents=True,exist_ok=True)
 if(folder/'completed.json').exists()and json.loads((folder/'completed.json').read_text()).get('version')==3:print('Already prepared',folder);return
 started=time.time();atomic_json(folder/'status.json',dict(status='loading',layer=a.layer))
 m=json.loads((ROOT/'experiment.json').read_text());events=Events(m,a.layer);model=QueryRouter(m,a.layer,'q64_mean8').cuda()
 items=events.calibration_items(30000+a.layer,12);qfeatures=[];kfeatures=[];qtargets=[];ktargets=[];ids=[];buckets=2**model.depth
 for item in items:
  assert item['row']['split']=='train'
  q,raw,_,_=events.gpu(item);k=torch.cat((raw['kh'],raw['kc']),1)
  qi=torch.linspace(0,q.shape[1]-1,256,device=q.device).long();ki=torch.linspace(0,k.shape[1]-1,256,device=k.device).long()
  qs=q[:,qi].float();ks=k[:,ki].float();ql=labels(q[:,qi].contiguous(),model.tree('q'));kl=labels(k.contiguous(),model.tree('k'))
  prob=torch.bmm(qs,k.float().transpose(1,2)).mul_(1/math.sqrt(128)).softmax(-1)
  qt=torch.zeros(q.shape[0],len(qi),buckets,device=q.device).scatter_add_(2,kl.long()[:,None].expand(-1,len(qi),-1),prob)
  selected=prob[:,:,ki];kt=torch.zeros(q.shape[0],buckets,len(ki),device=q.device).scatter_add_(1,ql.long()[:,:,None].expand(-1,-1,len(ki)),selected).transpose(1,2)
  kt=kt/kt.sum(-1,keepdim=True).clamp_min(1e-20)
  qfeatures.append(qs);kfeatures.append(ks);qtargets.append(qt.clamp_min(0).sqrt());ktargets.append(kt.clamp_min(0).sqrt());ids.append(item['row']['sample_id'])
  atomic_json(folder/'status.json',dict(status='collecting_dense_behavior',layer=a.layer,events=len(ids),total=len(items)))
  del q,k,raw,prob
 q=torch.cat(qfeatures,1);k=torch.cat(kfeatures,1);qt=torch.cat(qtargets,1);kt=torch.cat(ktargets,1)
 atomic_json(folder/'status.json',dict(status='fitting_behavior_trees',layer=a.layer))
 rawq,qfit=fit(q,qt,model.depth);rawk,kfit=fit(k,kt,model.depth)
 anchoredq,aqfit=fit(q,qt,model.depth,model.tree('q'));anchoredk,akfit=fit(k,kt,model.depth,model.tree('k'))
 pool=[x for x in events.train if x['row']['sample_id']not in ids];probes=random.Random(30000+a.layer).sample(pool,6)
 qtree,ktree,jointq,jointk,selection=choose_native(model,events,probes,anchoredq,anchoredk)
 # In-sample signature scores are diagnostics only; full96-event Val measures generalization.
 metrics={}
 for side,x,target,new,tile in [('q',q,qt,qtree,64),('k',k,kt,ktree,128)]:
  metrics[side]=dict(old_signature_tile_dispersion=signature_dispersion(x,target,model.tree(side),tile),new_signature_tile_dispersion=signature_dispersion(x,target,new,tile))
 save(folder/'trees.pt',dict(q_weight=qtree.weight.cpu(),q_threshold=qtree.threshold.cpu(),k_weight=ktree.weight.cpu(),k_threshold=ktree.threshold.cpu(),joint_q_weight=jointq.weight.cpu(),joint_q_threshold=jointq.threshold.cpu(),joint_k_weight=jointk.weight.cpu(),joint_k_threshold=jointk.threshold.cpu(),raw_q_weight=rawq.weight.cpu(),raw_q_threshold=rawq.threshold.cpu(),raw_k_weight=rawk.weight.cpu(),raw_k_threshold=rawk.threshold.cpu(),sample_ids=ids,split='train',reference='Opt29 frozen grouping',method='balanced Hellinger response partitions distilled to linear tree planes; anchored updates selected on held-out Train native loss'))
 atomic_json(folder/'selection.json',selection)
 atomic_json(folder/'fit.json',dict(q=qfit,k=kfit,anchored_q=aqfit,anchored_k=akfit,metrics=metrics,scope='Train calibration diagnostic only'))
 atomic_json(folder/'completed.json',dict(completed=True,version=3,layer=a.layer,sample_ids=ids,split='train',events=len(ids),tokens_per_event=256,depth=model.depth,seconds=time.time()-started,metrics=metrics,selection=selection))
 atomic_json(folder/'status.json',dict(status='completed',layer=a.layer,metrics=metrics));print('completed',a.layer,metrics,flush=True)
if __name__=='__main__':main()
