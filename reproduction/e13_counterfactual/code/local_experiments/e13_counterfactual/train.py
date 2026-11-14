import argparse,time
import numpy as np
import torch
from ..e12_swap_prediction.train import Net
from .core import *

def load(sp,regime='CF',exclude=None):
 ids=sorted(p.name for p in (OUT/'events').iterdir() if split(p.name)==sp and (exclude is None or p.name not in exclude));F=[];G=[];J=[];meta=[];counts=[]
 for eid in ids:
  d=np.load(OUT/'events'/eid/'states.npz');n=1 if regime=='Risk' else len(d['names']);counts.append(n)
  for s in range(n):F.append(d['features'][s]);G.append(d['gains'][s]);J.append(d['nmse'][s]);meta.append(dict(event_id=eid,state=s,name=str(d['names'][s])))
 return ids,np.stack(F),np.stack(G),np.array(J),meta,np.array(counts)
def event_normalizer(X,counts):
 start=0;means=[];squares=[]
 for n in counts:
  f=X[start:start+n];means.append(f.mean((0,1)));squares.append(np.square(f).mean((0,1)));start+=n
 mean=np.mean(means,0);std=np.sqrt(np.maximum(np.mean(squares,0)-mean**2,0));return mean,np.maximum(std,1e-6)
def infer(weights,X):
 w=np.load(weights);m=Net();m.load_state_dict({k:torch.from_numpy(w[k]) for k in ('fc1.weight','fc1.bias','fc2.weight','fc2.bias')});m.eval();a=(X-w['mean'])/w['std'];a[...,int(w['slots']):]=0
 with torch.no_grad():return m(torch.tensor(a,dtype=torch.float32)).numpy()
def metrics(scores,g,j):
 idx=scores.argmax(1);gain=g[np.arange(len(g)),idx];oracle=np.maximum(g.max(1),0);m=ranking_metrics(gain,j,oracle);m['best_action_hit_rate']=float(np.mean(idx==g.argmax(1)));return m

def fit(name,arch,regime,gpu,exclude=None):
 dest=OUT/'models'/name;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'training.json').exists():return
 torch.set_num_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dev=f'cuda:{gpu}'
 ids,F,g,j,meta,counts=load('train',regime,exclude);mean,std=event_normalizer(F,counts);slots=20 if arch=='A0' else 32
 X=(F-mean)/std;X[...,slots:]=0;x=torch.tensor(X,dtype=torch.float32,device=dev);norm=g/np.maximum(g.std(1,keepdims=True),1e-12);teacher=torch.softmax(torch.tensor(norm/.3,dtype=torch.float32,device=dev),-1)
 rng=np.random.default_rng(17);sample=rng.integers(0,counts,size=(2000,len(ids)))+np.r_[0,np.cumsum(counts)[:-1]][None];sample=torch.tensor(sample,device=dev)
 model=Net().to(dev);initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest();opt=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0);history=[];torch.cuda.synchronize(dev);start=time.monotonic()
 for step in range(2000):
  inds=sample[step];opt.zero_grad(set_to_none=True);score=model(x[inds]);loss=-(teacher[inds]*torch.log_softmax(score,-1)).sum(1).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step==0 or (step+1)%100==0:history.append(dict(step=step+1,loss=float(loss.detach())))
 torch.cuda.synchronize(dev);wall=time.monotonic()-start
 np.savez(dest/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()},mean=mean,std=std,slots=slots)
 validation={}
 for sp in ('train','validation'):
  eids,XX,GG,JJ,mm,cc=load(sp,'CF',exclude if sp=='train' else None);scores=infer(dest/'weights.npz',XX);np.savez_compressed(dest/(sp+'_scores.npz'),scores=scores);dump(dest/(sp+'_index.json'),mm)
  risk=np.array([m['state']==0 for m in mm]);validation[sp]=dict(risk=metrics(scores[risk],GG[risk],JJ[risk]),all_states_descriptive=metrics(scores,GG,JJ))
 dump(dest/'development.json',validation);dump(dest/'training.json',dict(name=name,arch=arch,regime=regime,seed=17,updates=2000,parameters=1089,event_count=len(ids),state_count=len(F),train_events=ids,counts=counts.tolist(),excluded_events=exclude,initial_weight_sha256=initial,sampler_sha256=hashlib.sha256(sample.cpu().numpy().tobytes()).hexdigest(),normalizer='equal event; equal state within event',loss=history,wall_seconds=wall,weights_sha256=sha(dest/'weights.npz')));print(name,validation['validation']['risk'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--arch',choices=['A0','A2'],required=True);a.add_argument('--regime',choices=['Risk','CF'],required=True);a.add_argument('--gpu',type=int,default=0);a.add_argument('--name');a.add_argument('--exclude-file');x=a.parse_args();fit(x.name or x.arch+'_'+x.regime,x.arch,x.regime,x.gpu,json.loads(Path(x.exclude_file).read_text()) if x.exclude_file else None)
