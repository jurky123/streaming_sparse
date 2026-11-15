from .core import *
from ..e17_swap_decomposition.evaluate import metrics

def run():
 lock=json.loads((OUT/'calibration_lock.json').read_text());lockhash=sha(OUT/'calibration_lock.json');performance=[];chosen=[];allenergies=[];alldirections=[];allinteractions=[];sourcechoices=[]
 for eid in ids():
  d=np.load(OUT/'events'/eid/'terms.npz');result=json.loads((OUT/'events'/eid/'result.json').read_text());ident=dict(event_id=eid,split=split(eid));G=d['gains'];J=d['nmse'];S=len(G);sels={'risk':np.array([0]),'post_risk':np.arange(1,S),'all':np.arange(S)}
  allenergies+=result['energy'];alldirections+=result['directions'];allinteractions+=result['interactions'];sourcechoices+=result['selected'];scores={}
  for model in lock['models']:
   rep=model['representation'];c=model['common_scale'];lam=model['E17_lambda'];ip=c*d[rep+'_unit_I'];bp=c*lam*d[rep+'_unit_B'];scores[rep+'__proxy_calibrated']=ip-bp
   for qn,factor in [('64',1),('16',4)]:
    I=d['I' if qn=='64' else 'I16']*factor;B=d['B' if qn=='64' else 'B16']*factor
    scores[rep+'__teacherI_Q'+qn]=I-bp;scores[rep+'__teacherB_Q'+qn]=ip-B
  scores['teacherIB_Q64']=d['I']-d['B'];scores['teacherIB_Q16']=4*(d['I16']-d['B16']);scores['full_oracle']=G
  for name,score in scores.items():
   for scope,sel in sels.items():performance.append(ident|dict(method=name,scope=scope,**metrics(score[sel],G[sel],J[sel])))
   choices=score.argmax(1)
   for i,j in enumerate(choices):chosen.append(ident|dict(method=name,state=str(d['names'][i]),state_index=i,action=int(j),remove=int(d['remove'][j]),add=int(d['add'][j]),execute=bool(score[i,j]>0),score=float(score[i,j]),gain=float(G[i,j]),nmse=float(J[i]),I=float(d['I'][i,j]),B=float(d['B'][i,j]),D=float(d['D'][i,j])))
 table('A_energy_per_state.csv',allenergies);table('B_interaction_per_state.csv',allinteractions);table('B_selected_sources.csv',sourcechoices);table('C_per_event.csv',performance);table('C_selected_actions.csv',chosen);table('D_directions_per_state.csv',alldirections)
 # Collapse within-event rows before computing equal-event population averages.
 def aggregate(rows,keys):
  first=[]
  for vals in sorted({tuple(r[k] for k in ['event_id']+keys) for r in rows}):
   rr=[r for r in rows if tuple(r[k] for k in ['event_id']+keys)==vals];v=dict(zip(['event_id']+keys,vals))
   for k in rr[0]:
    if k in v or k in ['state','selection','action','candidates']:continue
    a=[r[k] for r in rr if r.get(k) is not None]
    if a and isinstance(a[0],(int,float)):v[k]=float(np.mean(a))
   first.append(v)
  out=[]
  for vals in sorted({tuple(r[k] for k in keys) for r in first}):
   rr=[r for r in first if tuple(r[k] for k in keys)==vals];v=dict(zip(keys,vals));v['events']=len(rr)
   for k in rr[0]:
    if k in v or k=='event_id':continue
    a=[r[k] for r in rr if r.get(k) is not None]
    if a and isinstance(a[0],(int,float)):v[k]=float(np.mean(a))
   if 'oracle_gain' in v:
    v['pooled_gated_recovery']=v['gated_gain']/v['oracle_gain'] if v['oracle_gain']>1e-30 else None;v['pooled_positive_filtered_recovery']=v['positive_filtered_gain']/v['oracle_gain'] if v['oracle_gain']>1e-30 else None
   if 'total' in v:
    v['pooled_V_over_total']=v['V_energy']/v['total'] if v['total']>1e-30 else None;v['pooled_Z_over_total']=v['Z_energy']/v['total'] if v['total']>1e-30 else None;v['pooled_cross_over_total']=v['cross']/v['total'] if v['total']>1e-30 else None
   out.append(v)
  return out
 A=aggregate(allenergies,['split','level','scope']);B=aggregate(allinteractions,['split','scope','candidates']);C=aggregate(performance,['split','scope','method']);D=aggregate(alldirections,['split','level','scope','component'])
 for name,rows in [('A_energy_summary.csv',A),('B_interaction_summary.csv',B),('C_performance_summary.csv',C),('D_directions_summary.csv',D)]:table(name,rows)
 assert sha(OUT/'calibration_lock.json')==lockhash
 dump(OUT/'summary.json',dict(A=A,B=B,C=C,D=D,calibration_lock_sha=lockhash));print(json.dumps([r for r in C if r['split']=='evaluation' and r['scope']=='risk'],indent=2))
if __name__=='__main__':run()
