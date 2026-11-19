"""Independent normalized-mixture scoring/gradient and conservation checks."""
from .run import *
def run_audit():
 torch.set_num_threads(2);rows=[]
 for seed in [17,29]:assert (D/f's{seed}/complete.json').exists()
 for seed in [17,29]:
  bundle=bundle_for('C',seed);model=LocalReader().cuda().eval();model.load_state_dict(torch.load(E29/f's{seed}/selected.pt',map_location='cuda',weights_only=False)['model']);fit=torch.load(OUT/'fit.pt',weights_only=True);B=fit['B'].cuda().double();scale=json.loads((E29/f's{seed}/local_scale.json').read_text())['scale'];qs=torch.linspace(0,63,16,device='cuda').round().long()
  for split in ['train','development']:
   cd=CACHE/f'{split}_s{seed}';metas=json.loads((cd/'metadata.json').read_text())
   with (D/f's{seed}/decisions.csv').open() as fh:dec=[r for r in csv.DictReader(fh) if r['split']==split and r['policy']=='Set']
   worst=sorted(dec,key=lambda r:float(r['executed_gain']))[:2];ids={m['event']:i for i,m in enumerate(metas)}
   audit_ids=sorted(set([0,len(metas)//3,2*len(metas)//3]+[ids[r['event']] for r in worst]))
   for i in audit_ids:
    meta=metas[i];ev=torch.load(OUT/meta['event'],map_location='cpu',weights_only=True);sx=torch.load(cd/meta['summary_file'],map_location='cpu',weights_only=True);s=Summary(*(sx['summary'][k].cuda() for k in ['keys','mean','detail','stats']));q=ev['q'].cuda().float();cache=np.load(cd/('event_%04d.npz'%meta['index']));key='l%sh%s'%(ev['layer'],ev['head']);dense=ev['dense'].cuda().double()
    with torch.no_grad():
     c=contributions(q,s,bundle.layers[key].basis,ev['log_scale'].cuda().double());lm=c['logzh']-c['A'].log()[:,None];bt=ev['page_logz'].cuda().double()-ev['log_scale'].cuda().double()[:,None]-c['logzh'];ap=summary_output(c,s,bundle.layers[key].basis);rt=(ev['page_output'].cuda().double()-ap)@B;center=(ap-c['o'][:,None])@B;ff=torch.as_tensor(cache['F'],device='cuda').double();F=torch.cat([torch.ones_like(ff[...,:1]),ff],-1);rp,bp=model(s,F[qs]);mask,r,a=route(q[qs],s,bundle.layers[key],ev['own']);g=gains(c,ev['N'].cuda().double(),ev['Z'].cuda().double(),dense,mask,r,a)
    rr=rp.double().detach().requires_grad_();bb=bp.double().detach().requires_grad_();norm=dense.square().sum()/64+1e-12
    losses0,bs,gs,sc,ind=losses(rr,bb,rt[qs],bt[qs],c['mass'][qs],lm[qs],center[qs],mask,r,a,g,norm,scale,fit)
    score,independent=local_scores(rr,bb,lm[qs],center[qs],mask,r,a,norm)
    diff=float((score-sc).abs().max());assert torch.allclose(score,sc,rtol=1e-8,atol=1e-10)
    # Independently reconstruct the set objective from separately normalized outputs.
    target=torch.softmax(torch.cat([g,g.new_zeros(1)])/(.3*gs),0)
    rank=-(target*torch.log_softmax(torch.cat([score,score.new_zeros(1)])/gs,0)).sum()
    alt=torch.nn.functional.smooth_l1_loss(score/gs,g/gs)+.1*rank
    ga=torch.autograd.grad(losses0[2],(rr,bb),retain_graph=True);gb=torch.autograd.grad(alt,(rr,bb),retain_graph=True)
    de=max(float((u-v).abs().max()) for u,v in zip(ga,gb));assert all(torch.allclose(u,v,rtol=1e-7,atol=1e-9) for u,v in zip(ga,gb))
    truth=(lm[qs]+bt[qs]).softmax(-1);logpred=(lm[qs]+bb).log_softmax(-1);kl=(truth*(truth.clamp_min(1e-300).log()-logpred)).sum(-1).mean();kg=torch.autograd.grad(kl,bb)[0];expected=(logpred.exp()-truth)/16;ke=float((kg-expected).abs().max());assert ke<1e-12
    # Re-evaluate one full-Q64 real action through the actual mixed objective.
    j=int(g.argmax());mm=mask.clone();mm[r[j]]=False;mm[a[j]]=True
    assert int(mm.sum())==int(mask.sum()) and bool(mm[ev['own']])
    direct=objective(c,ev['N'].cuda().double(),ev['Z'].cuda().double(),dense,mask)-objective(c,ev['N'].cuda().double(),ev['Z'].cuda().double(),dense,mm)
    ge=float(abs(direct-g[j]));assert ge<1e-9
    jp=int(score.argmax());mp=mask.clone();mp[r[jp]]=False;mp[a[jp]]=True
    pred_direct=objective(c,ev['N'].cuda().double(),ev['Z'].cuda().double(),dense,mask)-objective(c,ev['N'].cuda().double(),ev['Z'].cuda().double(),dense,mp)
    ge=max(ge,float(abs(pred_direct-g[jp])));assert torch.allclose(pred_direct,g[jp],rtol=1e-5,atol=1e-9),(meta['event'],float(pred_direct),float(g[jp]))
    rows.append(dict(seed=seed,split=split,event=meta['event'],score_error=diff,set_gradient_error=de,kl_gradient_error=ke,true_gain_error=ge,proposal_gain_direct=float(pred_direct),proposal_gain_incremental=float(g[jp])))
 table(D/'independent_checks.csv',rows)
 # All partition shares must conserve their corresponding nonnegative signal.
 partition=['endpoint_pair','other_candidate','background'];n=0;max_partition_error=0.
 for seed in [17,29]:
  with (D/f's{seed}/output_gradient_roles.csv').open() as f:rs=list(csv.DictReader(f))
  groups=collections.defaultdict(list)
  for r in rs:
   if r['role'] in partition:groups[r['event']].append(r)
  for ev,rs in groups.items():
   for key in ['summary_mass_share','true_mass_share','predicted_mass_share','local_loss_share','beta_loss_share','beta_grad_l1_share','set_beta_grad_l1_share','kl_grad_l1_share']:
    v=[float(r[key]) for r in rs if r[key]]
    if v:
     max_partition_error=max(max_partition_error,abs(sum(v)-1));assert abs(sum(v)-1)<1e-6,(seed,ev,key,sum(v))
   n+=1
 dump(D/'audit.json',dict(passed=True,independent_events=len(rows),true_gain_max_absolute_difference=max(r['true_gain_error'] for r in rows),true_gain_tolerance=dict(rtol=1e-5,atol=1e-9),gain_note="Original incremental true gains suffer cancellation on one extreme audited action; direct positive-mixture recomputation confirms sign and magnitude. Original training labels retained.",partition_events=n,max_partition_sum_error=max_partition_error,partition_tolerance=1e-6,no_optimizer=True,test_accessed=False,raw_KV_read=False,limitations=['Parameter gradients are frozen final-checkpoint snapshots, not historic Adam updates.','Actual mass weighting and all mean reductions retained.','Global clipping cannot change component ratios within an event.']))
 print('AUDIT PASSED',len(rows),n)
if __name__=='__main__':run_audit()
