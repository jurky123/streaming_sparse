from .report import *
@torch.no_grad()
def run(seed):
 data=Data(seed);dest=D/f's{seed}';checks=[];p=read(dest/'pooling_counterfactual.csv');base=[r for r in p if r['bounded']=='True' and r['recenter']=='True'];assert len(base)==4 and all(float(r['baseline_key_max_error'])==0 for r in base)
 curves=read(dest/'fullset_curves.csv');assert len(curves)==2304 and all(np.isfinite(float(r['setloss'])) for r in curves)
 rows=read(dest/'trained_decisions.csv');assert len(rows)==2880
 for split in ['train','development']:
  r=min([x for x in rows if x['split']==split and x['policy']=='Set'],key=lambda x:float(x['executed_gain']));d=data.load(r['event']);ev=torch.load(OUT/r['event'],map_location='cpu',weights_only=True);key=f'l{ev["layer"]}h{ev["head"]}';c=contributions(ev['q'].cuda().float(),d['s'],data.bundle.layers[key].basis,ev['log_scale'].cuda().double());mask=d['mask'].clone()
  if r['execute']=='True':mask[int(r['remove'])]=False;mask[int(r['add'])]=True
  assert int(mask.sum())==int(d['mask'].sum()) and bool(mask[d['own']]);N,Z,od=[ev[k].cuda().double() for k in ['N','Z','dense']];direct=float(objective(c,N,Z,od,d['mask'])-objective(c,N,Z,od,mask));error=abs(direct-float(r['executed_gain']));assert np.isclose(direct,float(r['executed_gain']),rtol=1e-5,atol=1e-9);checks.append(dict(split=split,event=r['event'],gain_error=error))
 table(dest/'audit_checks.csv',checks);dump(dest/'audit.json',dict(passed=True,updates=2304,baseline_pooling_exact=True,Test=False,checks=checks));print('AUDIT PASS',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
