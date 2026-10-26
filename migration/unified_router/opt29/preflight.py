import json,torch,gc,time
from .setup import ROOT,VARIANTS
from .model import QueryRouter
from .data import Events
from ..common import atomic_json,paired_seed
from ..opt16.grouping import gather
from ..opt28.probe import validate
from ..opt12.precision import clip_grad_norm
@torch.no_grad()
def verify_kernel(d,o):
 # Independent FP32 reference for every representative of three real tiles.
 errors=[]
 for h in [0,5,11]:
  for t in [0,d['bar'].shape[1]//2,d['bar'].shape[1]-1]:
   ids=d['candidates'][h,t].long();key=d['kv'].state['key'][h,ids].float();lc=d['kv'].state['logcount'][h,ids];bar=d['bar'][h,t].float()
   ll=(bar@key.reshape(-1,64).T).reshape(o['r'],len(ids),16)/(128**.5)+lc[None];ref=ll.logsumexp(-1);got=d['fine'][h,t]
   err=float((ref-got).abs().max());assert err<.005,err;errors.append(err)
 return max(errors)
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;paired_seed(29024);m=json.loads((ROOT/'experiment.json').read_text());events=Events(m,24);item=events.train[0];q,raw,truth,targets=events.gpu(item);rows=[]
 for name in VARIANTS:
  t0=time.time();model=QueryRouter(m,24,name).cuda();prev=events.previous(item,model.options['refresh']);out,d=model.deployed(q,raw,targets,prev);assert torch.isfinite(out).all();assert float(d['executed'])<.22
  fine_err=verify_kernel(d,model.options);y,info=model(q,raw,targets,prev);assert torch.equal(out,y.detach());loss=(y-truth).square().sum()/truth.square().sum();loss.backward();gn=clip_grad_norm(model.trainable(),1.)
  qgrad=float(model.query_weight.grad.norm())if model.query_weight.grad is not None else None
  assert gn>0 and(model.options['mode']!='learned' or qgrad>0)
  opt=torch.optim.AdamW(model.trainable(),lr=1e-5);opt.step();assert all(torch.isfinite(p).all()for p in model.trainable())
  row=dict(variant=name,loss=float(loss),gradient_norm=float(gn),query_gradient=qgrad,fine_max_abs=fine_err,executed=float(d['executed']),seconds=time.time()-t0);rows.append(row);print(row,flush=True);atomic_json(ROOT/'preflight.json',dict(passed=False,rows=rows))
  del model,prev,out,d,y,info,loss,opt;gc.collect();torch.cuda.empty_cache()
 atomic_json(ROOT/'preflight.json',dict(passed=True,rows=rows,real_forward_equal=True,gradients_finite=True,sequence_query_capture_complete=True))
if __name__=='__main__':main()
