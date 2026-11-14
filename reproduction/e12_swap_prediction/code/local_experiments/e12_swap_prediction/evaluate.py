import json,time,csv
import numpy as np
import torch
from .core import *
from .train import Net,load_states,decision_metrics

def lock():
 path=OUT/'evaluation_lock.json'
 if path.exists():return json.loads(path.read_text())
 gate=json.loads((OUT/'development_gate.json').read_text());names=[f'A{i}_s17' for i in range(3)];winner=gate['winner'];multi=False
 if winner:
  names.append(f'{winner}_s29');second=json.loads((OUT/'models'/names[-1]/'selection.json').read_text());multi=bool(second['selected']['gate'])
 models=[]
 for name in names:
  model=OUT/'models'/name;sel=json.loads((model/'selection.json').read_text());models.append(dict(name=name,weights_sha256=sha(model/'weights.npz'),threshold=sel['selected']['threshold'],selection_sha256=sha(model/'selection.json'),validation=sel['selected']))
 result=dict(frozen_at=time.time(),models=models,multi_enabled=multi,multi_architecture=winner if multi else None,reason='both-seed validation gate' if multi else 'selected architecture did not pass both seeds; no multiround or expanded-state training',primary_comparison='A0/A1/A2 seed17',secondary='winner seed29 only, selected using development',test='horse_field72; no threshold/model updates after this file')
 dump(path,result);return result

def model_scores(name,F):
 m=np.load(OUT/'models'/name/'weights.npz');net=Net();net.load_state_dict({k:torch.from_numpy(m[k]) for k in ('fc1.weight','fc1.bias','fc2.weight','fc2.bias')});net.eval();x=(F-m['mean'])/m['std'];x[...,int(m['slots']):]=0
 with torch.no_grad():return net(torch.tensor(x,dtype=torch.float32)).numpy()
def run():
 torch.set_num_threads(1);locked=lock();ids,F,g,j=load_states('evaluation');rows=[];summary=[];oracle=np.maximum(g.max(1),0);summary.append(dict(name='Risk',calibration='baseline',**metric(np.zeros(len(j)),j,oracle)));summary.append(dict(name='Medium_oracle',calibration='true_Q64_single_swap',**metric(oracle,j,oracle)))
 for m in locked['models']:
  name=m['name'];assert sha(OUT/'models'/name/'weights.npz')==m['weights_sha256'];scores=model_scores(name,F);threshold=float('inf') if m['threshold'] is None else m['threshold'];np.savez_compressed(OUT/'models'/name/'evaluation_scores.npz',event_ids=np.array(ids),scores=scores)
  for label,t in [('frozen',threshold),('threshold0',0)]:
   stats,dec=decision_metrics(scores,g,j,t);summary.append(dict(name=name,calibration=label,threshold=None if np.isinf(t) else t,**stats))
   for i,eid in enumerate(ids):
    statefile=np.load(OUT/'states'/f'{eid}.npz');action=int(dec['chosen'][i]);take=bool(dec['take'][i]);r=int(statefile['remove'][action]);a=int(statefile['add'][action]);x=statefile['mask'].copy()
    if take:x[r]=False;x[a]=True
    assert x[int(statefile['own'])] and x.sum()==int(statefile['K'])
    rows.append(dict(event_id=eid,name=name,calibration=label,threshold=None if np.isinf(t) else t,no_op=not take,remove=r if take else None,add=a if take else None,score=float(dec['confidence'][i]),risk_nmse=float(j[i]),nmse=float(j[i]-dec['gain'][i]),gain=float(dec['gain'][i]),gain_pct=float(100*dec['gain'][i]/j[i]),pool_oracle_gain=float(oracle[i]),pool_recovery=float(dec['gain'][i]/oracle[i]) if oracle[i]>1e-12 else None,selected=np.flatnonzero(x).tolist()))
 dump(OUT/'evaluation_summary.json',summary);dump(OUT/'evaluation_per_instance.json',rows)
 with (OUT/'evaluation_per_instance.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
