"""CPU independent counterfactual check; GPU native parity is a queue prerequisite."""
import torch,json,math
from types import SimpleNamespace
from . import actions
from .setup import ROOT
def main():
 torch.set_num_threads(2);torch.manual_seed(32);h,n,dim=2,1032,128;k=128;g=64;b=2*math.ceil(n/k);t=math.ceil(n/g)
 q=torch.randn(h,n,dim);keys=torch.randn(h,n*2,dim);values=torch.randn_like(keys);p=torch.rand(h,t,b).softmax(-1);ids=p.topk(5,-1).indices;mask=torch.zeros_like(p,dtype=torch.bool).scatter_(-1,ids,True);ix=torch.arange(n)[None].expand(h,-1).int()
 d=dict(p=p,mask=mask,ix=ix,kv=SimpleNamespace(k=keys,v=values),seg=n,g=g,k=k);a=actions.candidates(d);targets=torch.tensor([0,62,100,700,1031]);lab=actions.labels(q,targets,d,a);errors=[]
 for index in [0,1,9,17,80]:
  choice=torch.full((h,t),index,dtype=torch.long);valid=a['valid'][...,index];choice=torch.where(valid,choice,0);selected=actions.apply(d,a,choice)
  original_mask=[]
  for chunk in range(2):original_mask.append(selected[...,chunk*math.ceil(n/k):(chunk+1)*math.ceil(n/k)].repeat_interleave(k,-1)[...,:n])
  token_mask=torch.cat(original_mask,-1)[lab['head'],lab['group']];ll=q[:,targets]@keys.transpose(1,2)/math.sqrt(dim);out=ll.masked_fill(~token_mask,-torch.inf).softmax(-1)@values;truth=ll.softmax(-1)@values;ref=(out-truth).square().sum(-1);err=float((ref-lab['errors'][...,index]).abs().max());assert err<1e-4,err;errors.append(err)
  if index>=17:assert torch.equal(selected.sum(-1),mask.sum(-1))
 basis=torch.linalg.qr(torch.randn(h,128,64)).Q;model=SimpleNamespace(routing={'key_projection':basis});vb=basis[...,:32]
 kcontent=actions.content(d,model,'K32',vb);vcontent=actions.content(d,model,'V32',vb);kx,kg=actions.features(d,a,kcontent);vx,vg=actions.features(d,a,vcontent)
 assert torch.isfinite(kx).all()and torch.isfinite(vx).all()
 changed=dict(d,kv=SimpleNamespace(k=keys,v=values+torch.randn_like(values)))
 k2,g2=actions.features(changed,a,actions.content(changed,model,'K32',vb));v2,gv2=actions.features(changed,a,actions.content(changed,model,'V32',vb))
 assert torch.equal(kx,k2)and torch.equal(kg,g2),'V leakage in K-only online features'
 assert not torch.equal(vx,v2),'V-aware features fail to read V'
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'math_check.json').write_text(json.dumps(dict(passed=True,max_counterfactual_abs=max(errors),keep_add_drop_swap_checked=True,tail_pages_excluded=True,k_features_invariant_to_v=True,v_features_sensitive_to_v=True),indent=2));print('passed',max(errors))
if __name__=='__main__':main()
