import json,torch,math,gc
from .setup import ROOT
from .model import QueryRouter
from .pipeline import Pipeline
from .data import Events
from ..opt16.pipeline import GroupedRouter
from ..opt16.grouping import gather
from ..opt8.projection import project
from ..opt28.probe import validate
from ..common import atomic_json
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((ROOT/'experiment.json').read_text());events=Events(m,24);item=next(x for x in events.val if x['row']['step']==3 and x['row']['history_tokens']==14040);q,raw,_,_=events.gpu(item);rows=[]
 for name in ['q32_a8','q64_a4','q64_a8','q64_a16','q64_a8_pair','q64_a8_early','q64_mean8','q64_sample8']:
  model=QueryRouter(m,24,name).cuda().eval().requires_grad_(False);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False);pipe=Pipeline(model,raw);prev=events.previous(item,model.options['refresh']);y,d=pipe.run(q,raw['kc'],raw['vc'],True,previous_q=prev);o=model.options;g,r=o['g'],o['r'];h,n,_=q.shape;t=math.ceil(n/g);ix=d['q_order'].long();z=d['projected'];valid=torch.arange(t*g,device=q.device).reshape(t,g)<n
  if o['mode']=='learned':
   source=project(prev,d['kv'].state['basis']).float()if prev is not None else z.float();refa=(torch.bmm(source,model.query_weight.half().float().transpose(1,2))+model.query_bias[:,None]).softmax(-1).half()
   if prev is None:refa=torch.zeros_like(refa).scatter_(1,ix[...,None].expand(-1,-1,r),refa)
   assignment_error=float((refa.float()-pipe.assignments.float()).abs().max());assert assignment_error<.001,assignment_error
   a=torch.gather(pipe.assignments,1,ix[...,None].expand(-1,-1,r)).float();a=torch.nn.functional.pad(a,(0,0,0,t*g-n)).reshape(h,t,g,r)
  elif o['mode']=='mean':a=torch.nn.functional.one_hot(torch.arange(g,device=q.device)//(g//r),r).float()[None,None].expand(h,t,-1,-1)*valid[None,:,:,None];assignment_error=0.
  else:a=None;assignment_error=0.
  zz=torch.nn.functional.pad(z.float(),(0,0,0,t*g-n)).reshape(h,t,g,64)
  if a is not None:
   count=a.sum(2);bar=(torch.einsum('htgr,htgd->htrd',a,zz)/count.clamp_min(1e-20)[...,None]).half();mass=count/valid.sum(-1)[None,:,None]
  else:
   pos=torch.minimum(torch.arange(t,device=q.device)[:,None]*g+torch.arange(r,device=q.device)*(g//r)+(g//r)//2,torch.tensor(n-1,device=q.device));bar=z[:,pos];mass=torch.full((h,t,r),1/r,device=q.device)
  error=float((bar.float()-d['bar'].float()).square().sum()/bar.float().square().sum().clamp_min(1e-20));assert error<1e-6,error;assert torch.allclose(mass,d['mass'],atol=1e-6,rtol=1e-6)
  kernel_error=validate(pipe.lib,gather(q,d['q_order']),d['selection'],d['q_order'],y);rows.append(dict(variant=name,assignment_max_abs=assignment_error,aggregation_nmse=error,native_max_nmse=kernel_error));print(rows[-1],flush=True)
  del model,pipe,y,d,prev;gc.collect();torch.cuda.empty_cache()
 atomic_json(ROOT/'aggregation_check.json',dict(passed=True,rows=rows,independent_fp32_reference=True))
if __name__=='__main__':main()
