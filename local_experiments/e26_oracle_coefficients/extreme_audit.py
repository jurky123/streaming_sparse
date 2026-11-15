"""Long-double audit of the large-score cases, without removing any event."""
from .core import *
@torch.no_grad()
def run():
 torch.set_num_threads(2);fit=torch.load(OUT/'fit.pt',weights_only=True);B=fit['B'].cuda();qs=torch.linspace(0,63,16,device='cuda').round().long();out=[]
 for seed,indices in [(17,[47,182,231]),(29,[239,230,190])]:
  bc=bundle_for('C',seed)
  for idx in indices:
   d=load(files('development')[idx]);s=bc.make(d,'C');c=contributions(d['q'],s,bc.layers[d['key']].basis,d['log_scale']);tz,_,tb,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);lm=c['logzh']-c['A'].log()[:,None];F=features(bc.reader,d['q'],c);z,b,_,_=fit_coeff(F,lm,tz,tb,fit['zscale'],fit['bscale']);mask,r,a=route(d['q'][qs],s,bc.layers[d['key']],d['own']);norm=float(d['dense'][qs].square().sum()/16+1e-12);saved=np.load(E26/f's{seed}'/f'panel_{idx:04d}.npz')
   zz=z[qs].cpu().numpy().astype(np.longdouble);logs=lm[qs].cpu().numpy().astype(np.longdouble);beta=b[qs].cpu().numpy().astype(np.longdouble);mass=np.exp(logs);exact=np.exp(logs+beta);target=zz.sum(1)/exact.sum(1)[:,None]
   def objective(x):
    den=np.where(x[None],exact,mass).sum(1);o=zz[:,x].sum(1)/den[:,None];return np.sum((target-o)**2,axis=-1).mean()/np.longdouble(norm)
   for st_idx,state in enumerate(saved['states']):
    if st_idx==0:rr=r.cpu().numpy();aa=a.cpu().numpy();ref=saved['initial_OracleCoeff_Cphi']
    else:rr=saved['remove'];aa=saved['add'];ref=saved['OracleCoeff_Cphi'][st_idx]
    base=objective(state);vals=[]
    for rem,add in zip(rr,aa):
     x=state.copy();x[rem]=False;x[add]=True;vals.append(base-objective(x))
    vals=np.array(vals,dtype=np.longdouble);j=int(vals.argmax());jr=int(ref.argmax());relative=float(np.max(abs(vals-ref))/(1+np.max(abs(vals))));same_action=j==jr;same_value=abs(vals[j]-vals[jr])<=np.longdouble('1e-10')*max(1,abs(vals[j]));out.append(dict(seed=seed,event=str(files('development')[idx].relative_to(OUT)),state=st_idx,max_abs_predicted_gain=float(max(abs(vals))),max_score_scaled_error=relative,same_argmax=same_action,equivalent_tie=bool(same_value),reference_best_gain=float(vals[j]),fp64_selected_longdouble_gain=float(vals[jr]),same_noop=bool(vals[j]>0)==bool(ref.max()>0)));assert relative<1e-8 and (same_action or same_value) and out[-1]['same_noop'],out[-1]
 dump(E26/'extreme_precision_audit.json',dict(longdouble_bits=np.finfo(np.longdouble).nmant,rows=out,events=6,states=len(out),test_accessed=False,note='All events retained; high predicted gains are not erased as numerical outliers.'))
 print('EXTREME AUDIT PASSED',len(out))
if __name__=='__main__':run()
