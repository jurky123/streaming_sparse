from .core import *
import argparse

def run(eid,device):
 out=OUT/'stage1'/eid;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 ctx=load(eid,device);results=[];scores={};saved={};audit={}
 for form in ['original','centered']:
  ep,zp,meta=direct(ctx,form);r=evaluate(ctx,form,ep,zp,meta=meta);scores.update(r.pop('scores'));results.append(r);saved[form]={'e':ep.cpu(),'z':zp.cpu()}
  if form=='original':
   old=np.load(E19/'moments'/eid/'scores.npz');err=max(float(np.max(abs(scores['original__all_pages__'+p+'__Q'+str(M)]-old[p+'__M2__Q'+str(M)]))) for p in ctx['panels'] for M in [16,64]);assert err<1e-9;audit['E19_M2_max_score_error']=err
 torch.save(saved,out/'predictions.pt');np.savez_compressed(out/'scores.npz',**scores);dump(out/'result.json',dict(event_id=eid,methods=results,audit=audit,snapshot_sha=sha(DATA/'representatives'/f'{eid}.pt'),bases_sha=sha(OUT/'bases.npz')));print(eid,[(r['performance'][0]['method'],r['performance'][0]['gated_relative_pct']) for r in results])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--event',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.event,a.device)
