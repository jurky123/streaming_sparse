import argparse,time,json
import numpy as np
import torch
from .core import *
SLOTS={'A0':20,'A1':24,'A2':32}
class Net(torch.nn.Module):
 def __init__(self):
  super().__init__();self.fc1=torch.nn.Linear(32,32);self.fc2=torch.nn.Linear(32,1)
 def forward(self,x):return self.fc2(torch.nn.functional.gelu(self.fc1(x))).squeeze(-1)
def load_states(sp):
 ids=sorted(p.stem for p in (OUT/'states').glob('*.npz') if split(p.stem)==sp);ds=[dict(np.load(OUT/'states'/f'{e}.npz')) for e in ids]
 return ids,np.stack([d['features'] for d in ds]),np.stack([d['gains'] for d in ds]),np.array([float(d['baseline_nmse']) for d in ds])
def decision_metrics(scores,g,j,threshold):
 best=scores.argmax(1);conf=scores[np.arange(len(scores)),best];take=conf>threshold;gain=np.where(take,g[np.arange(len(g)),best],0);oracle=np.maximum(g.max(1),0);m=metric(gain,j,oracle);positive=g>1e-12;pred=scores>threshold;tp=(positive&pred).sum(1);fp=(~positive&pred).sum(1);pos=positive.sum(1);m.update(no_op=int((~take).sum()),swapped=int(take.sum()),swap_positive_precision=float(np.mean(gain[take]>1e-12)) if take.any() else None,positive_action_recall=float(np.mean(tp/np.maximum(pos,1))),positive_action_precision=float(np.mean(tp/np.maximum(tp+fp,1))),false_positive_action_rate=float(np.mean(fp/np.maximum((~positive).sum(1),1))))
 return m,dict(chosen=best,take=take,gain=gain,confidence=conf,oracle=oracle)
def select(scores,g,j):
 trials=[]
 for threshold in [0,.1,.25,.5,1,2,4,8,16,float('inf')]:
  m,_=decision_metrics(scores,g,j,threshold);trials.append(dict(threshold=None if np.isinf(threshold) else threshold,**m))
 eligible=[r for r in trials if r['degradation_rate']<=.15];best=max(eligible,key=lambda r:(r['mean_gain_pct'],-r['degradation_rate'],-r['swapped']))
 return best,trials

def train(arch,seed,gpu):
 out=OUT/'models'/f'{arch}_s{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'selection.json').exists():return
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;device='cuda:'+str(gpu)
 ids,X,g,j=load_states('train');vids,V,vg,vj=load_states('validation');assert len(ids)==len(vids)==36
 mean=X.mean((0,1));std=X.std((0,1));std=np.maximum(std,1e-6)
 def tensor(a):
  a=(a-mean)/std;a[...,SLOTS[arch]:]=0;return torch.tensor(a,dtype=torch.float32,device=device)
 x=tensor(X);v=tensor(V);scale=np.maximum(g.std(1),1e-12);norm=g/scale[:,None];teacher=torch.softmax(torch.tensor(np.column_stack((norm,np.zeros(len(g))))/.3,dtype=torch.float32,device=device),-1)
 net=Net().to(device);initial={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()};assert sum(p.numel() for p in net.parameters())==1089
 opt=torch.optim.Adam(net.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0);history=[];torch.cuda.synchronize(device);start=time.monotonic()
 for step in range(1,2001):
  opt.zero_grad(set_to_none=True);scores=net(x);logits=torch.cat((scores,torch.zeros((len(x),1),device=device)),1);loss=-(teacher*torch.log_softmax(logits,-1)).sum(1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  if step==1 or step%100==0:history.append(dict(step=step,loss=float(loss.detach())))
 torch.cuda.synchronize(device);seconds=time.monotonic()-start
 with torch.no_grad():ts=net(x).cpu().numpy();vs=net(v).cpu().numpy()
 selection,trials=select(vs,vg,vj);raw,_=decision_metrics(vs,vg,vj,0)
 np.savez(out/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()},mean=mean,std=std,slots=SLOTS[arch],seed=seed)
 np.savez_compressed(out/'development_predictions.npz',train_scores=ts,validation_scores=vs,train_ids=np.array(ids),validation_ids=np.array(vids),teacher_probabilities=teacher.cpu().numpy())
 dump(out/'training.json',dict(architecture=arch,seed=seed,gpu=gpu,parameters=1089,updates=2000,wall_seconds=seconds,train_states=ids,validation_states=vids,loss=history,initial_weight_sha256=hashlib.sha256(b''.join(v.tobytes() for v in initial.values())).hexdigest(),weights_sha256=sha(out/'weights.npz')))
 dump(out/'selection.json',dict(architecture=arch,seed=seed,selected=selection,threshold0=raw,trials=trials,weights_sha256=sha(out/'weights.npz'),evaluation_accessed=False));print(arch,seed,selection,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--arch',choices=list(SLOTS),required=True);a.add_argument('--seed',type=int,default=17);a.add_argument('--gpu',type=int,default=0);x=a.parse_args();train(x.arch,x.seed,x.gpu)
