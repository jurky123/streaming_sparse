from .train import *
import re
@torch.no_grad()
def run():
 state=torch.load(PRE,weights_only=False,map_location='cpu')['model'];count=0
 for group in ['A_matched','B','C']:
  for seed in [17,29]:
   ck=torch.load(OUT/'training'/f'{group}_s{seed}'/'selected.pt',weights_only=False,map_location='cpu')
   for key,t in ck['layers'].items():
    l,h,n=re.match(r'l(\d+)h(\d+)\.(.*)',key).groups();h=int(h)
    if n in ['pool_k','pool_v','pool_feature'] and group!='A_matched':continue
    ref=state[f'layers.{l}.{n}'];ref=ref if n=='logtheta' else ref[h:h+1];assert torch.equal(t,ref);count+=1
 b=Bundle();ck=torch.load(OUT/'training/C_s17/selected.pt',weights_only=False,map_location='cuda');b.layers.load_state_dict(ck['layers']);b.reader.load_state_dict(ck['reader']);qs=torch.linspace(0,63,16,device='cuda').round().long();err=0.
 for p in files('development')[:3]:
  d=load(p);m=b.layers[d['key']];s=b.make(d,'C');mask,r,a=route(d['q'][qs],s,m,d['own']);scores=[]
  for shift in [0.,-20.,20.]:
   c=contributions(d['q'][qs],s,m.basis,d['log_scale'][qs]+shift);z,u,beta=b.reader(d['q'][qs],s,c);scores.append(predicted(z,u,c['mass'],mask,r,a,beta))
  for sc in scores[1:]:err=max(err,float((sc-scores[0]).abs().max()));assert torch.allclose(sc,scores[0],atol=1e-10,rtol=1e-8)
 dump(OUT/'frozen_and_scale_audit.json',dict(passed=True,frozen_tensors_checked=count,teacher_numerical_scale_invariance_max_error=err,scope='same readout under common Query exponent reference shifts; no Dense or exact page contributions used by Reader'))
if __name__=='__main__':run()
