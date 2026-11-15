from .core import *

def run():
 torch.set_num_threads(2);torch.manual_seed(17)
 # Differentiate the five-dimensional solve independently of the BF16 STE.
 F=torch.randn(6,3,5,dtype=torch.double,device='cuda',requires_grad=True);lm=torch.randn(6,3,dtype=torch.double,device='cuda');target=torch.randn(6,3,16,dtype=torch.double,device='cuda')
 # loss's fixed64*16 floor is just a constant in this synthetic gradient check.
 ok=torch.autograd.gradcheck(lambda x:variable_projection_loss(x,lm,target,.053),F,eps=1e-6,atol=1e-5,rtol=1e-3);assert ok
 b=bundle_for('C',17)
 for p in b.parameters():p.requires_grad_(False)
 net=AdaptivePhi().cuda();fit=torch.load(OUT/'fit.pt',weights_only=True);opt=torch.optim.Adam(net.parameters(),lr=1e-4);checks=[]
 for idx in [0,47]:
  d,s,c,z,bt,lm=prepare(b,files('train')[idx]);F=net(d['q'],s);loss=variable_projection_loss(F,lm,z,fit['zscale']);opt.zero_grad();loss.backward();norms={n:float(p.grad.norm()) for n,p in net.named_parameters()};assert all(math.isfinite(v) for v in norms.values());assert norms['query.weight']>0 and norms['page.2.weight']>0;torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step();checks.append(dict(event=str(files('train')[idx].relative_to(OUT)),loss=float(loss),gradients=norms))
 dump(E27/'stage0.json',dict(gradcheck=bool(ok),real_training_checks=checks,no_stage0_weights_reused=True,test_accessed=False));print('STAGE0 PASS')
if __name__=='__main__':run()
