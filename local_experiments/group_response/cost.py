"""Paired costs on the SAME actual clean chunk, independent of rollout timing."""
import json,torch,statistics
from pathlib import Path
from ..broad_common import Engine,ROOT as DATA
from ..review_v1.model import Model
from ..review_v1.ops import SummaryOps,Config
from ..review_v1.runtime import PageStore
from .core import Runtime,grouping,GROUPS,responses
from .run import ROOT,BASE,PRE
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 engine=Engine();engine.wrapper.requires_grad_(False);model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False);rt=Runtime(engine,model,'G-KVR','Dense');captured=[];old=rt.group
 def group(l,k,v):
  if engine.current['chunk']==5 and l in [1,15,29]:captured.append((l,k[0].transpose(0,1).clone(),v[0].transpose(0,1).clone()))
  return old(l,k,v)
 rt.group=group
 emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda();z=torch.load(ROOT/'video/horse_field_s101/G-KVR/Dense/latents.pt',map_location='cuda',weights_only=True)
 for c in range(6):engine.forward(z[:,c*3:c*3+3],emb,c,4)
 rows=[]
 for l,k,v in captured:
  m=model[l];before={n:p.detach().clone() for n,p in m.named_parameters()};key,vr=responses(m,k.float()[:,None],v.float()[:,None]);key_only,_=responses(m,k.float()[:,None],v.float()[:,None],need_value=False);_,vr_only=responses(m,k.float()[:,None],v.float()[:,None],need_key=False);assert torch.equal(key,key_only) and torch.equal(vr,vr_only)
  cache={'k':torch.empty(1,410*64,12,128,device='cuda',dtype=torch.bfloat16)};cache['v']=torch.empty_like(cache['k']);store=PageStore(cache,Config())
  # Interleaved trials, all candidates see exactly the same clean KV and weights.
  trials={g:[] for g in GROUPS}
  for trial in range(13):
   for g in GROUPS[trial%4:]+GROUPS[:trial%4]:
    a,b,c,d=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    a.record();kg,vg,order=grouping(m,k,v,g);b.record();s=SummaryOps.build(m,kg,vg);c.record();store.commit(kg,vg,s,order,5*4608);d.record();d.synchronize()
    if trial>=3:trials[g].append(dict(group_ms=a.elapsed_time(b),summary_ms=b.elapsed_time(c),write_ms=c.elapsed_time(d),commit_total_ms=a.elapsed_time(d)))
  assert all(torch.equal(before[n],p) for n,p in m.named_parameters())
  for g,values in trials.items():rows.append(dict(layer=l,group=g,calls=len(values),**{n:statistics.mean(x[n] for x in values) for n in values[0]},trials=values))
 rt.close();(ROOT/'paired_group_cost.json').write_text(json.dumps(dict(status='complete',branch_features_bitwise_equal=True,parameters_unchanged=True,scope='same real horse_field seed101 Dense G-KVR clean chunk5; layers1/15/29; 3 warmups and 10 interleaved trials; GPU event elapsed; G-K skips unused value branch',rows=rows),indent=2));print('COST COMPLETE',flush=True)
if __name__=='__main__':main()
