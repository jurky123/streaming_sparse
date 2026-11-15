from .core import *
def main():
 torch.set_num_threads(2);checks=[]
 for seed in [17,29]:
  torch.manual_seed(seed);bundle=bundle_for('C',seed);phi=load_phi(seed)
  for p in itertools.chain(bundle.parameters(),phi.parameters()):p.requires_grad_(False)
  reader=LocalReader().cuda();old=torch.load(E27/f'B_s{seed}'/'initial.pt',weights_only=True,map_location='cpu');assert all(torch.equal(v.cpu(),old[n]) for n,v in reader.state_dict().items())
  for idx in [0,143]:
   d,s,c,rt,bt,lm,center=prepare_local(bundle,files('train')[idx]);F=phi(d['q'],s);qs=torch.linspace(0,63,16,device='cuda').round().long();mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);rp,bp=reader(s,F[qs]);z=reconstruct(rp,bp,lm[qs],center[qs]);sc,_=safe_predictions(z,bp,lm[qs],mask,r,a,1.);loss=(c['mass'][qs][...,None]*(rp-rt[qs])).square().mean()+sc.square().mean();reader.zero_grad();loss.backward();gn=sum(float(p.grad.double().square().sum()) for p in reader.parameters());assert math.isfinite(gn) and gn>0
   zero=torch.zeros_like(bp);z0=reconstruct(torch.zeros_like(rp),zero,lm[qs],center[qs]);sc0,_=safe_predictions(z0,zero,lm[qs],mask,r,a,1.);assert torch.equal(z0,torch.zeros_like(z0)) and torch.equal(sc0,torch.zeros_like(sc0))
   _,_,cc,zt,_,_=prepare(bundle,files('train')[idx]);truth=reconstruct(rt,bt,lm,center);err=float((truth-zt).abs().max()/zt.abs().max());assert err<1e-7
   checks.append(dict(seed=seed,train_idx=idx,identity_relative=err,gradient_squared_norm=gn,zero_correction_noop=True,parameters=sum(p.numel() for p in reader.parameters()),matched_initial=True))
 # Small double-precision analytic coupling gradient check, independent of network.
 r=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True);b=torch.randn(2,3,dtype=torch.float64,requires_grad=True)*.1;lm=torch.log_softmax(torch.randn(2,3,dtype=torch.float64),-1);c=torch.randn(2,3,4,dtype=torch.float64)
 passed=torch.autograd.gradcheck(lambda r,b:reconstruct(r,b,lm,c),(r,b));assert passed;dump(E29/'stage0.json',dict(checks=checks,analytic_gradcheck=passed,test_accessed=False));print('STAGE0 DONE',flush=True)
if __name__=='__main__':main()
