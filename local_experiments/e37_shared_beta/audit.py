from .run import *
from .report import table
import hashlib

def arr(t):return t.double().cpu().numpy().astype(np.longdouble)
def softmax(x):
 w=np.exp(x-x.max(1,keepdims=True));return w/w.sum(1,keepdims=True)
def main():
 qs=np.linspace(0,63,16).round().astype(int);checks=[];fits=[];hashes={};replay=[];old=list(csv.DictReader((E30/'decision_per_event.csv').open()));oldmap={(int(x['seed']),x['event'],x['method'],x['policy']):x for x in old}
 for seed in [17,29]:
  dest=DEST/f's{seed}';rows=json.loads((dest/'decisions.json').read_text());sources=json.loads((dest/'sources.json').read_text());byevent=collections.defaultdict(list)
  for r in rows:byevent[r['event']].append(r)
  worst=max(rows,key=lambda x:-x['executed_gain'])['event']
  for idx,src in enumerate(sources):
   pp=E30/f's{seed}/panel_{idx:04d}.npz';xp=E30/f's{seed}/beta_{idx:04d}.pt';event=src['event'];p=np.load(dest/f'panel_{idx:04d}.npz');original=np.load(pp);x=torch.load(xp,map_location='cpu',weights_only=True)
   for f,k in [(pp,'panel_sha'),(xp,'beta_sha'),(OUT/event,'event_sha')]:
    h=sha(f);assert h==src[k];hashes[str(f.relative_to(ROOT))]=h
   for k in ['states','remove','add','initial_remove','initial_add','true_gain','true_initial_gain']:assert np.array_equal(p[k],original[k])
   C=np.zeros(len(p['states'][0]),bool);C[p['initial_remove']]=True;C[p['initial_add']]=True;lm=arr(x['logmass'])[qs];pred=arr(x['betas']['Predicted'])[qs];true=arr(x['beta_true'])[qs];d=true-pred
   def mean(ix):
    ww=np.exp(2*(lm[:,ix]-lm[:,ix].max(1,keepdims=True)));return (ww*d[:,ix]).sum(1)/ww.sum(1)
   b,bc,bb=mean(np.ones_like(C)),mean(C),mean(~C);err=max(float(np.max(np.abs(v-p[k]))) for v,k in [(b,'shared'),(bc,'candidate'),(bb,'background')]);assert err<1e-8;fits.append(dict(seed=seed,event=event,max_offset_error=err));betas={'E29':pred,'Shared-Q':pred+b[:,None],'CandidateBackground-Q':pred+np.where(C[None],bc[:,None],bb[:,None]),'TrueBeta':true}
   for row in byevent[event]:
    m=p['states'][0];new=m.copy();r,a=row['remove'],row['add'];own=len(m)-72+row['tile'];assert m[r] and not m[a] and r!=own
    if row['execute']:new[r]=False;new[a]=True
    assert new.sum()==int(.1*len(m)) and new[own];ii=np.flatnonzero((p['initial_remove']==r)&(p['initial_add']==a));assert len(ii)==1;assert row['proposal_gain']==float(p['true_initial_gain'][ii[0]])
    if row['method'] in ['E29','TrueBeta']:
     z=oldmap[seed,event,'Predicted' if row['method']=='E29' else 'True',row['policy']];assert row['execute']==(z['execute']=='True') and r==int(z['remove']) and a==int(z['add']);replay.append(abs(row['executed_gain']-float(z['executed_gain'])))
   if idx not in [0,47,143,144,230,287] and event!=worst:continue
   ev=torch.load(OUT/event,weights_only=True);norm=float(ev['dense'][qs].square().sum()/16+1e-12);cen=arr(x['center'])[qs];rp=arr(x['predicted_response'])[qs];v=cen+rp;m=p['states'][0]
   for method,beta in betas.items():
    le=lm+beta;target=(softmax(le)[...,None]*v).sum(1)
    def hybrid(mm):return (softmax(np.where(mm[None],le,lm))[...,None]*np.where(mm[None,:,None],v,cen)).sum(1)
    for row in [r for r in byevent[event] if r['method']==method and r['policy']=='Set']:
     new=m.copy();new[row['remove']]=False;new[row['add']]=True;H=hybrid(m);delta=hybrid(new)-H;sc=(2*(target-H)*delta-delta*delta).sum(-1).mean()/norm;error=abs(float(sc)-row['predicted_gain']);assert error<1e-8+1e-6*abs(float(sc));checks.append(dict(seed=seed,event=event,method=method,selected_score_error=error))
    mx=le.max(1,keepdims=True);logD=mx+np.log(np.exp(le-mx).sum(1,keepdims=True));extra=max(0,(lm-logD).max());net=np.exp(le-logD-extra)[...,None]*(v-target[:,None])-np.exp(lm-logD-extra)[...,None]*(cen-target[:,None]);energy=(net*net).sum(-1).mean(0);ind=energy[p['initial_add']]-energy[p['initial_remove']];err=float(np.max(np.abs(ind-p['independent_'+method])))/max(1,float(np.max(np.abs(ind))));assert err<1e-10
  done=json.loads((dest/'complete.json').read_text());assert done['protocol_sha']==sha(DEST/'protocol.json')
  prior=json.loads((ROOT/f'artifacts/e36_query_coverage_v1/A_s{seed}/complete.json').read_text())
  for path,key in [(checkpoint('C',seed),'C_sha'),(E29/f's{seed}/selected.pt','reader_sha')]:assert sha(path)==prior[key];hashes[str(path.relative_to(ROOT))]=sha(path)
 for f in (ROOT/'local_experiments/e37_shared_beta').glob('*.py'):hashes[str(f.relative_to(ROOT))]=sha(f)
 hashes[str((DEST/'protocol.json').relative_to(ROOT))]=sha(DEST/'protocol.json');table('source_hashes.csv',[dict(path=k,sha256=v) for k,v in sorted(hashes.items())]);dump(DEST/'audit.json',dict(events=576,decisions=len(rows)*2,LS_checks=fits,longdouble_score_checks=checks,baseline_replay_count=len(replay),baseline_max_error=max(replay),frozen_models_unchanged=True,raw_kv_loaded=False,training=False,test=False));print('AUDIT PASSED',len(fits),len(checks),max(replay))
if __name__=='__main__':main()
