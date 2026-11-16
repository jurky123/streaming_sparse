from .reference import *
import csv

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run():
 torch.set_num_threads(2);tests=[];rng=np.random.default_rng(4901)
 # Fixed analytic/random tests, no real gain or threshold search.
 for i in range(24):
  P=4;rr=torch.tensor(rng.normal(size=(1,P,2)),device='cuda',dtype=torch.double);bb=torch.tensor(rng.normal(size=(1,P))*[1,10,100,650][i%4],device='cuda',dtype=torch.double);lm=torch.tensor(rng.normal(size=(1,P))*3,device='cuda',dtype=torch.double).log_softmax(-1);c=torch.tensor(rng.normal(size=(1,P,2)),device='cuda',dtype=torch.double);mask=torch.tensor([True,False,True,False],device='cuda');lo,hi=cert(rr,bb,lm,c,mask,0,1,False);val=exact_scores(*[x.cpu().numpy() for x in [rr,bb,lm,c]],mask.cpu().tolist(),np.array([0]),np.array([1]),[0],800)[0]
  with mp.workdps(850):assert mp.mpf(float(lo))<=mp.mpf(val)<=mp.mpf(float(hi))
  tests.append(dict(test=i,low=lo,high=hi,reference=val,contained=True))
 table(D/'interval_self_checks.csv',tests);points=[];source=[]
 for seed in [17,29]:
  data=Data(seed);paths0={'E29':E29/f's{seed}/selected.pt','T0':E39/f'T0_s{seed}/epoch1.pt','Full_Pset':E48/f's{seed}/fullset_final.pt'};models={n:model_from(p).eval() for n,p in paths0.items()};refs=[]
  for kind in ['tails','all']:
   p=D/f'reference_{seed}_{kind}.csv';assert p.exists();refs+=read(p)
  seen=set()
  for r in refs:
   key=(r['model'],r['event'])
   if key in seen:continue
   seen.add(key);d=data.load(r['event'])
   with torch.no_grad():
    rp,bp=models[r['model']](d['s'],d['F']);sc,_=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);j=int(sc.argmax());ms=d['mask'].clone();ms[d['r'][j]]=False;ms[d['a'][j]]=True;le=d['lm']+bp
    def out(mask):return (torch.where(mask[None],le,d['lm']).softmax(-1)[...,None]*(d['center']+mask[None,:,None]*rp)).sum(1)
    H=out(d['mask']);Hn=out(ms);ww=torch.where(d['mask'][None],le,d['lm']).softmax(-1);Hsame=torch.bmm(ww[:,None],d['center'])[:,0]+torch.bmm((ww*d['mask'][None])[:,None],rp.double())[:,0];DD=(le.softmax(-1)[...,None]*(d['center']+rp)).sum(1)
    for q in range(16):points.append(dict(seed=seed,model=r['model'],event=r['event'],probe=int(data.qs[q]),H_norm=float(H[q].norm()),Hnew_norm=float(Hn[q].norm()),delta_norm_FP64=float((Hn[q]-H[q]).norm()),E_norm=float((DD[q]-H[q]).norm()),legacy_noop_delta_norm=float((Hsame[q]-H[q]).norm()),legacy_noop_rawgain=float(2*((DD[q]-H[q])*(Hsame[q]-H[q])).sum()-(Hsame[q]-H[q]).square().sum()),unified_noop_delta_norm=float((out(d['mask'])[q]-H[q]).norm())))
  for part in range(4):
   dest=D/f's{seed}_part{part}';done=json.loads((dest/'complete.json').read_text());assert done['events']==360;source.append(done)
  # Historical original decisions must match saved baseline results exactly.
  history={'E29':ROOT/f'artifacts/e41_training_signal_v1/s{seed}/decisions.csv','T0':ROOT/f'artifacts/e42_local_coverage_v1/T0_s{seed}/decisions.csv','Full_Pset':E48/f's{seed}/trained_decisions.csv'};orig=[]
  for part in range(4):orig+=read(D/f's{seed}_part{part}/decisions.csv')
  for name,p in history.items():
   old={(x['event'],x['policy']):x for x in read(p)};rr=[x for x in orig if x['model']==name and x['protocol']=='original'];assert len(rr)==2880
   for x in rr:
    y=old[(x['event'],x['policy'])];assert x['execute']==y['execute'] and abs(float(x['executed_gain'])-float(y['executed_gain']))<1e-9
  data.check_frozen()
 table(D/'output_change_reference.csv',points);dump(D/'audit.json',dict(passed=True,events_per_seed=1440,models=3,interval_self_tests=len(tests),historical_replay_rows=17280,weights_unchanged=True,Test=False,training=False,conditional_precision_guarantee='fixed binary inputs, IEEE754 operations outward, MPFR-directed exp; interval inclusion also tested with independent mpmath reference'))
 print('AUDIT PASSED',flush=True)
if __name__=='__main__':run()
