from .core import *

def run():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.manual_seed(17);state=torch.load(PRE,map_location='cpu',weights_only=False)['model'];results=[]
 B=torch.from_numpy(np.load(ROOT/'artifacts/e19_response_moments_v1/pca16.npz')['net']).float().cuda() # implementation check only; stage1 refits on TRAIN
 for eid in ['city_walk_s101_c5_l15_s0_h8_t61','horse_field_s101_c5_l15_s0_h8_t61']:
  snap=torch.load(OLD/'representatives'/f'{eid}.pt',map_location='cuda',weights_only=True);q=snap['q'].float();k=snap['k'][None];v=snap['v'][None];m=layer(state,15,8,'cuda');reader=Reader().cuda();hp=338;own=399;qs=torch.linspace(0,63,16,device='cuda').round().long()
  with torch.no_grad():
   logit=torch.einsum('qd,ptd->qpt',q.double(),k[0].double())/math.sqrt(128);scale=logit.flatten(1).amax(1);w=(logit-scale[:,None,None]).exp();Z=w.sum(-1).float();N=torch.einsum('qpt,ptd->qpd',w,v[0].double()).float();od=N.sum(1)/Z.sum(1)[:,None];scale=scale.float()
  s=summary(m,k,v,hp);matches={n:bool(torch.equal(getattr(s,n),snap['summary'][n])) for n in ['keys','mean','detail','stats']};relative={n:float((getattr(s,n).float()-snap['summary'][n].float()).square().sum()/(snap['summary'][n].float().square().sum()+1e-20)) for n in matches};assert max(relative.values())<1e-5,relative
  opt=torch.optim.Adam(list(p for p in m.parameters() if p.requires_grad)+list(reader.parameters()),lr=1e-4);trace=[];oldtargets=None;gradient={}
  for it in range(4):
   s=summary(m,k,v,hp);c=contributions(q,s,m.basis,scale);cp={n:x[qs] for n,x in c.items()};mask,r,a=route(q[qs],s,m,own);zstar,ustar,bs,t=labels(c,N,Z,od,B);exact_identity=c['o']+t[:,mask].sum(1)/(1+ustar[:,mask].sum(1))[:,None];err=float((exact_identity-mixed(c,N,Z,mask)).abs().max());assert err<2e-5
   z,u,beta=reader(q[qs],s,cp);pg=predicted(z,u,cp['mass'],mask,r,a);truth=gains(c,N,Z,od,mask,r,a);gain_scale=truth.detach().std().clamp_min(1e-6);zg=zstar[qs].detach().square().mean().sqrt().clamp_min(1e-6);bg=bs[qs].detach().square().mean().sqrt().clamp_min(1e-6)
   approx=objective(c,N,Z,od,mask);res=torch.nn.functional.smooth_l1_loss(z/zg,zstar[qs].detach()/zg)+torch.nn.functional.smooth_l1_loss(beta/bg,bs[qs].detach()/bg);target=torch.softmax(torch.cat([truth.detach(),truth.new_zeros(1)])/(.3*gain_scale),0);rank=-(target*torch.log_softmax(torch.cat([pg,pg.new_zeros(1)])/gain_scale,0)).sum();ls=torch.nn.functional.smooth_l1_loss(pg/gain_scale,truth.detach()/gain_scale)+.1*rank
   if it==0:
    gg=torch.autograd.grad(ls,[m.pool_k,m.pool_v,m.pool_feature],retain_graph=True);gradient={n:float(g.norm()) for n,g in zip(['pool_k','pool_v','pool_feature'],gg)};assert all(np.isfinite(x) and x>0 for x in gradient.values());oldtargets=(zstar.detach().clone(),truth.detach().clone())
   loss=approx+.1*res+.1*ls;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(m.parameters())+list(reader.parameters()),1,error_if_nonfinite=True);opt.step();trace.append(dict(iteration=it,approx=float(approx),res=float(res),set_loss=float(ls),identity_error=err))
  refreshed=labels(contributions(q,summary(m,k,v,hp),m.basis,scale),N,Z,od,B)[0];delta=float((refreshed-oldtargets[0]).abs().max());assert delta>0
  results.append(dict(event_id=eid,snapshot_sha=sha(OLD/'representatives'/f'{eid}.pt'),baseline_summary_exact_match=matches,baseline_summary_relative_error=relative,set_loss_pool_gradients=gradient,refreshed_target_max_change=delta,trace=trace,scope='implementation-only4updates, not trained model or generalization'))
 dump(OUT/'stage0.json',dict(passed=True,results=results));print('stage0 passed',results)
if __name__=='__main__':run()
