from .core import *
from ..e41_training_signal.run import parameter_audit

def run(seed):
 torch.set_num_threads(2);data,states,groups,initial=build(seed,verify=True);model=model_from(E29/f's{seed}/selected.pt');rows=[];grads=[];teachers=[];arrays={};teacher_sc={name:[] for name in ['M16B16','M16B128','Q64B16','Q64B128']}
 for i in range(0,96,2):
  d=states[i];m=d['meta'];rp,bp=outputs(model,d);ls,bs,gs,_,_=losses(rp,bp,d['rtrue'],d['btrue'],d['mass'],d['lm'],d['center'],d['mask'],d['r'],d['a'],d['g'],d['norm'],data.scale,data.fit)
  grads+=parameter_audit(model,ls,dict(seed=seed,event=m['event'],layer=m['layer']))
  bg=torch.autograd.grad(ls[1],bp,retain_graph=True)[0];sr,sb=torch.autograd.grad(ls[2],(rp,bp));phi=d['F'][...,1:]
  # Project beta error onto per-query common and centered pieces; compare gradients, not just SSE.
  err=bp.detach().double()-d['btrue'];common=err.mean(1,keepdim=True);gg=bg.double();gc=gg.mean(1,keepdim=True).expand_as(gg);rows.append(dict(seed=seed,event=m['event'],layer=m['layer'],phi_near99_fraction=float((phi.abs()>=.99).double().mean()),phi_exact1_fraction=float((phi.abs()==1).double().mean()),phi_mean_derivative=float((1-phi.square()).mean()),bs_mean=float(bs.mean()),bs_median=float(bs.median()),beta_loss=float(ls[1]),local_loss=float(ls[0]),set_loss=float(ls[2]),quality_output_gradnorm=float(bg.norm()),set_beta_output_gradnorm=float(sb.norm()),common_error_sse_fraction=float((common.square().sum()*err.shape[1])/err.square().sum()),common_quality_gradient_energy_fraction=float(gc.square().sum()/gg.square().sum())))
  if m['layer']!=29:continue
  ev=torch.load(OUT/m['event'],map_location='cpu',weights_only=True);key='l%sh%s'%(m['layer'],m['head']);q=ev['q'].cuda().float();c=contributions(q,d['s'],data.bundle.layers[key].basis,ev['log_scale'].cuda().double());lm=c['logzh']-c['A'].log()[:,None];bt=ev['page_logz'].cuda().double()-ev['log_scale'].cuda().double()[:,None]-c['logzh'];ap=summary_output(c,d['s'],data.bundle.layers[key].basis);rt=ev['page_output'].cuda().double()-ap;center=ap-c['o'][:,None]
  for stateidx in [i,i+1]:
   st=states[stateidx]
   for name in teacher_sc:
    qs=data.qs if name.startswith('M16') else torch.arange(64,device='cuda');B=data.B if name.endswith('B16') else torch.eye(128,device='cuda',dtype=torch.double)
    with torch.no_grad():sc,_=local_scores(rt[qs]@B,bt[qs],lm[qs],center[qs]@B,st['mask'],st['r'],st['a'],st['norm'])
    teacher_sc[name].append(sc.cpu().numpy());j=int(sc.argmax());ex=bool(sc[j]>0)
    if name=='Q64B128':assert torch.allclose(sc,st['g'],rtol=1e-5,atol=1e-8),(m['event'],float((sc-st['g']).abs().max()))
    teachers.append(dict(seed=seed,event=m['event'],state=stateidx,method=name,initial=st['initial'],oracle_gain=max(0.,float(st['g'].max())),executed_gain=float(st['g'][j]) if ex else 0.,execute=ex,proposal_gain=float(st['g'][j]),hindsight_gain=max(0.,float(st['g'][j]))))
 table(D/f's{seed}/physics.csv',rows);table(D/f's{seed}/physics_gradients.csv',grads);table(D/f's{seed}/layer29_teachers.csv',teachers);np.savez_compressed(D/f's{seed}/layer29_scores.npz',**{k:np.stack(v) for k,v in teacher_sc.items()});dump(D/f's{seed}/physics_complete.json',dict(seed=seed,events=48,layer29_events=16,teacher_states=32,Q64_full_closure=True,test=False));print('PHYSICS DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
