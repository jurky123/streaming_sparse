from .run import *
import mpmath as mp

def exact_scores(rp,bp,lm,c,mask,rs,aa,js,dps):
 with mp.workdps(dps):
  M,P,K=rp.shape;tot={j:mp.mpf(0) for j in js}
  for q in range(M):
   l=[mp.mpf(float(x)) for x in lm[q]];b=[mp.mpf(float(x)) for x in bp[q]];v=[[mp.mpf(float(c[q,p,k]))+mp.mpf(float(rp[q,p,k])) for k in range(K)] for p in range(P)];cc=[[mp.mpf(float(c[q,p,k])) for k in range(K)] for p in range(P)]
   def output(ms):
    log=[l[p]+(b[p] if ms[p] else 0) for p in range(P)];sh=max(log);w=[mp.exp(z-sh) for z in log];z=mp.fsum(w);return [mp.fsum(w[p]*(v[p][k] if ms[p] else cc[p][k]) for p in range(P))/z for k in range(K)]
   dense=output([True]*P);H=output(mask);E=[dense[k]-H[k] for k in range(K)]
   for j in js:
    ms=list(mask);ms[int(rs[j])]=False;ms[int(aa[j])]=True;Hnew=output(ms);delta=[Hnew[k]-H[k] for k in range(K)];tot[j]+=mp.fsum(2*E[k]*delta[k]-delta[k]**2 for k in range(K))/M
  return {j:mp.nstr(g,dps) for j,g in tot.items()}

def run(seed,near=False):
 torch.set_num_threads(2);data=Data(seed);sources={'E29':E29/f's{seed}/selected.pt','T0':E39/f'T0_s{seed}/epoch1.pt','Full_Pset':E48/f's{seed}/fullset_final.pt'};models={k:model_from(p).eval() for k,p in sources.items()};selected=[]
 tails=['data/forest_run_s702/c8_l29_s0_h9_t39.pt','data/reef_fish_s703/c5_l29_s0_h9_t37.pt']
 for name in models:
  for e in [tails[0] if seed==17 else tails[1]]:selected.append((name,e,'known_tail'))
 if near:
  selected=[]
  refs=[]
  for part in range(4):
   with (D/f's{seed}_part{part}/proposal_reference.csv').open() as f:refs+=list(csv.DictReader(f))
  for name in ['Full_Pset']:
   for split in ['train','development']:
    rr=[r for r in refs if r['model']==name and r['split']==split and float(r['raw'])>0]
    for desc,chosen in [('nearzero',min(rr,key=lambda x:abs(float(x['raw'])))),('positive_control',min(rr,key=lambda x:x['event']))]:selected.append((name,chosen['event'],desc))
 else:selected=selected[:]
 rows=[];point=[]
 for name,event,scope in selected:
  d=data.load(event)
  with torch.no_grad():rp,bp=models[name](d['s'],d['F']);sc,_=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);uni,inc,no,fb=paths(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'])
  old=sc*d['norm'];order=torch.argsort(old,descending=True,stable=True);js=list(dict.fromkeys([int(order[0]),int(order[1]),int(uni.argmax()),int(inc.argmax())]));end=torch.cat([d['lm'][:,d['r'][js]],d['lm'][:,d['a'][js]],(d['lm']+bp)[:,d['r'][js]],(d['lm']+bp)[:,d['a'][js]]],1);span=float(torch.maximum(d['lm'].max(1).values,(d['lm']+bp).max(1).values).max()-end.min());dps=max(180,int(2*span/np.log(10))+100)
  arrays=[x.detach().cpu().double().numpy() for x in [rp,bp,d['lm'],d['center']]];mask=d['mask'].cpu().tolist();rr=d['r'].cpu().numpy();aa=d['a'].cpu().numpy();first=exact_scores(*arrays,mask,rr,aa,js,dps);second=exact_scores(*arrays,mask,rr,aa,js,dps+80)
  for j in js:
   lo,hi=cert(rp,bp,d['lm'],d['center'],d['mask'],d['r'][j],d['a'][j],False)
   with mp.workdps(dps+100):
    value=mp.mpf(second[j]);v1=mp.mpf(first[j]);assert mp.mpf(float(lo))<=value<=mp.mpf(float(hi));stable=mp.sign(value)==mp.sign(v1);assert stable
   rows.append(dict(seed=seed,model=name,event=event,scope=scope,action=j,selected=j==int(order[0]),raw_fp64=float(old[j]),unified_fp64=float(uni[j]),increment_fp64=float(inc[j]),mp_dps=dps,mp_gain=first[j],mp_gain_check=second[j],mp_sign=int(mp.sign(mp.mpf(second[j]))),interval_low=float(lo),interval_high=float(hi),real_gain=float(d['g'][j]),interval_contains_reference=True))
  j=int(order[0]);le=d['lm']+bp;ms=d['mask'].clone();ms[d['r'][j]]=False;ms[d['a'][j]]=True;w0=torch.where(d['mask'][None],le,d['lm']).softmax(-1);w1=torch.where(ms[None],le,d['lm']).softmax(-1)
  for q in range(16):point.append(dict(seed=seed,model=name,event=event,scope=scope,probe=int(data.qs[q]),remove=int(d['r'][j]),add=int(d['a'][j]),predicted_beta_remove=float(bp[q,d['r'][j]]),predicted_beta_add=float(bp[q,d['a'][j]]),current_remove_mass=float(w0[q,d['r'][j]]),new_add_mass=float(w1[q,d['a'][j]]),logmass_remove=float(d['lm'][q,d['r'][j]]),logmass_add=float(d['lm'][q,d['a'][j]])))
  table(D/f'reference_{seed}_{"all" if near else "tails"}.csv',rows);table(D/f'endpoint_{seed}_{"all" if near else "tails"}.csv',point);print('REF',seed,name,scope,event,dps,flush=True)
 print('REFERENCE DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--near',action='store_true');a=p.parse_args();run(a.seed,a.near)
