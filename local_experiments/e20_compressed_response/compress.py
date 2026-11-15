from .core import *
import argparse

def run(eid,device):
 out=OUT/'compression'/eid;out.mkdir(parents=True,exist_ok=True)
 ctx=load(eid,device);lock=json.loads((OUT/'form_lock.json').read_text());assert lock['bases_sha']==sha(OUT/'bases.npz');form=lock['chosen'];results=[];scores={};audit={}
 def add(name,e,z,meta=None,history=False):
  for scope in (['all_pages','history_only'] if history else ['all_pages']):
   r=evaluate(ctx,name,e,z,scope,meta);scores.update(r.pop('scores'));results.append(r)
 e,z,m=direct(ctx,form);add('full_M2',e,z,m,True)
 for k in [4,8]:
  e,z,m=direct(ctx,form,ctx['bases']['key'+str(k)]);add('K_only'+str(k),e,z,m)
 for v in ['current8','pca8','pca16']:
  e,z,m=direct(ctx,form,uv=ctx['bases'][v]);add('V_only_'+v,e,z,m)
 for name,k,v in CONFIGS:
  uk=ctx['bases']['key'+str(k)];uv=ctx['bases'][v];stats=build_stats(ctx,form,uk,uv,False);e,z,m=from_stats(ctx,form,uk,uv,stats);ed,zd,_=direct(ctx,form,uk,uv);err=max(float((e-ed).abs().max()),float((z-zd).abs().max()));assert err<1e-10;audit[name+'_direct_error']=err;add(name+'_FP64',e,z,m)
  stats={n:t.bfloat16() for n,t in stats.items()};torch.save({n:t.cpu() for n,t in stats.items()},out/(name+'_BF16.pt'));b=sum(t.numel()*t.element_size() for t in stats.values())//410;assert b==8*(128+(1+uv.shape[1])*(k+k*(k+1)//2));audit[name+'_bytes_per_page']=b
  e,z,m=from_stats(ctx,form,uk,uv,stats);add(name+'_BF16',e,z,m,True)
 from ..e12_swap_prediction.core import objective
 checks=0
 for rr in results:
  for a in rr['selected']:
   mask=ctx['raw']['risk'].copy()
   if a['execute']:mask[a['remove']]=False;mask[a['add']]=True
   assert mask.sum()==int(ctx['raw']['K']) and mask[int(ctx['raw']['own'])]
   gain=ctx['J']-objective(ctx['raw'],mask);assert abs(gain-(a['true_gain'] if a['execute'] else 0))<1e-10;checks+=1
 audit['selected_masks_independently_checked']=checks
 np.savez_compressed(out/'scores.npz',**scores);dump(out/'result.json',dict(event_id=eid,form=form,lock_sha=sha(OUT/'form_lock.json'),methods=results,audit=audit));print(eid,'done',audit,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--event',required=True);p.add_argument('--device',required=True);a=p.parse_args();run(a.event,a.device)
