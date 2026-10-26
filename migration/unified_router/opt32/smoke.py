"""Real replay GPU check using an existing pilot checkpoint, never training output."""
import json,torch
from .setup import ROOT,PARENT,ANCHOR
from .worker import Policy,batch,decide
from . import actions
from ..opt30.model import QueryRouter
from ..opt30.data import Events
from ..opt16.pipeline import GroupedRouter
from ..opt5.probe import capture
from ..common import atomic_json,sha
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((PARENT/'experiment.json').read_text());events=Events(m,24);base=QueryRouter(m,24,ANCHOR).cuda().eval().requires_grad_(False);path=PARENT/f'training/layer24/{ANCHOR}/best.pt';base.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['model']);base.pipeline=GroupedRouter(base.dictionary(),base.tree('q'),base.tree('k'),compile=False);basis=base.value_basis[...,:32];rows=[]
 for kind in ['K32','V32','V8']:
  q,raw,truth,targets,pipe,d,a,x,gain=batch(base,events,events.train[0],kind,basis)
  with torch.no_grad():
   out=actions.attend(pipe,d,d['mask']);ref=pipe.attention(d['qi'],d['qs'],d['ids'],d['ix'],q.dtype);assert torch.equal(out,ref)
   for index in [1,9,17]:
    if bool(a['valid'][...,index].all()):
     mask=actions.apply(d,a,torch.full(d['mask'].shape[:2],index,device='cuda',dtype=torch.long));count=int(mask[0,0].sum());assert bool((mask.sum(-1)==count).all());ids=torch.where(mask,torch.arange(mask.shape[-1],device='cuda'),mask.shape[-1]).sort(-1).values[...,:count].int();assert torch.equal(actions.attend(pipe,d,mask),pipe.attention(d['qi'],d['qs'],ids,d['ix'],q.dtype))
   lab=actions.labels(q,targets,d,a);assert torch.isfinite(lab['errors']).all()
  norm=dict(mean=x.mean((1,2)),std=x.std((1,2),unbiased=False).clamp_min(.001));policy=Policy(norm).cuda();options=dict(learned=True,mode='swap');logits,chosen=decide(policy,x,gain,a,options);prob=(logits[lab['head'],lab['group']]/.1).softmax(-1);loss=(prob*lab['errors']).sum()/truth.square().sum();loss.backward();gn=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);assert torch.isfinite(gn)and gn>0
  torch.optim.AdamW(policy.parameters(),lr=1e-4).step()
  with torch.no_grad():
   def fn():
    dd=actions.context(pipe,base,q,raw);aa=actions.candidates(dd);xx,gg=actions.features(dd,aa,actions.content(dd,base,kind,basis));_,cc=decide(policy,xx,gg,aa,options);return actions.attend(pipe,dd,actions.apply(dd,aa,cc))
   expected=fn();graph,result=capture(fn);graph.replay();torch.cuda.synchronize();assert torch.equal(expected,result)
  row=dict(content=kind,base_native_exact=True,swap_add_drop_native_exact=True,graph_exact=True,gradient_norm=float(gn),counterfactual_finite=True);rows.append(row);print(row,flush=True)
 atomic_json(ROOT/'gpu_smoke.json',dict(passed=True,rows=rows,checkpoint=str(path),sha256=sha(path),scope='Existing L24 pilot checkpoint only; future study waits for complete frozen all29 foundation'))
if __name__=='__main__':main()
