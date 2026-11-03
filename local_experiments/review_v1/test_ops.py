"""Numerical contracts, CPU runnable; not a speed benchmark."""
import unittest
from types import SimpleNamespace
import torch
from .ops import Config,SummaryOps,RouteOps,AttentionOps,haar

class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192);torch.set_num_threads(2)
        self.k=torch.randn(2,5,64,128).bfloat16();self.v=(torch.randn_like(self.k.float())+2).bfloat16()
        self.m=SimpleNamespace(bank=torch.randn(2,32,128)*.03,pool_k=torch.zeros(2,4,128),pool_v=torch.zeros(2,4,128),pool_feature=torch.zeros(2,4,32),basis=torch.linalg.qr(torch.randn(2,128,8)).Q)
        self.s=SummaryOps.build(self.m,self.k,self.v,cheap=True)
        self.q=torch.randn(2,4,128).bfloat16()
    def test_zero_query_and_bytes(self):
        out,z=AttentionOps.evaluate(self.q*0,self.k,self.v,self.s,self.m.basis,torch.zeros(2,4,5,dtype=torch.bool))
        torch.testing.assert_close(out,self.s.mean.float().mean(1)[:,None].expand_as(out),atol=1e-6,rtol=1e-6)
        self.assertEqual(self.s.bytes(),2*5*1424)
        torch.testing.assert_close(z,torch.full_like(z,torch.tensor(320.).log()))
    def test_all_selected_and_residual_identity(self):
        allmask=torch.ones(2,4,5,dtype=torch.bool)
        od,zd=AttentionOps.evaluate(self.q,self.k,self.v,self.s,self.m.basis,allmask)
        dense=torch.nn.functional.scaled_dot_product_attention(self.q.float(),self.k.flatten(1,2).float(),self.v.flatten(1,2).float())
        torch.testing.assert_close(od,dense,atol=2e-6,rtol=2e-6)
        mask=torch.rand(2,4,5)>.5
        out,z=AttentionOps.evaluate(self.q,self.k,self.v,self.s,self.m.basis,mask)
        e,l,a,s=AttentionOps.pages(self.q,self.k,self.v,self.s,self.m.basis)
        residual=(l-zd[...,None]).exp()[...,None]*(e-od[:,:,None])-(s-zd[...,None]).exp()[...,None]*(a-od[:,:,None])
        rhs=(residual*(~mask)[...,None]).sum(2)*(zd-z).exp()[...,None]
        torch.testing.assert_close(od-out,rhs,atol=2e-6,rtol=2e-5)
        labels=AttentionOps.residual_labels(self.q,self.k,self.v,self.s,self.m.basis,od,zd)
        torch.testing.assert_close(labels,residual.double().norm(dim=-1),atol=2e-7,rtol=2e-5)
    def test_risk_zero_and_valid(self):
        valid=torch.tensor([[1,1,1,0,0],[1,0,1,1,0]],dtype=torch.bool)
        f,z=RouteOps.features(self.q*0,self.s,valid,block=2)
        torch.testing.assert_close(f[...,1:],torch.zeros_like(f[...,1:]))
        expected=self.s.stats[:,:,0,11].float()*valid/valid.sum(-1,keepdim=True)
        torch.testing.assert_close(f[...,0],expected[:,None].expand(2,4,5))
        self.assertTrue(torch.isfinite(z).all())
    def test_counts_overflow_and_drop(self):
        cfg=Config();scores=torch.rand(2,3,410);own=torch.tensor([[338,339,340]]).expand(2,-1)
        s=RouteOps.select(scores,own,cfg,threshold=-1)
        self.assertTrue((s['counts']==103).all());self.assertTrue((s['overflow']==307).all())
        self.assertTrue(s['mask'].gather(-1,own[...,None]).all())
        self.assertTrue((s['ids'][...,103:]==-1).all())
        self.assertTrue((s['mask'].sum(-1)==s['counts']).all())
        self.assertEqual(cfg.drop,22);self.assertEqual(Config(history_pages=360).drop,0)
        self.assertEqual(Config(history_pages=360).page_order('cpu').unique().numel(),72)
    def test_selected_attention_and_padding(self):
        from .runtime import SelectedAttention
        q=self.q[:,0:1].expand(-1,64,-1).contiguous()
        mask=torch.zeros(2,1,5,dtype=torch.bool);mask[:,:,1]=True;mask[:,:,3]=True
        ids=torch.tensor([[[1,3,-999,-999]],[[3,1,-999,-999]]],dtype=torch.int32)
        sel=dict(mask=mask,ids=ids,counts=torch.full((2,1),2,dtype=torch.int32))
        got=SelectedAttention.run(q,self.k,self.v,self.s,self.m.basis,sel)
        batched=SelectedAttention.run(q,self.k,self.v,self.s,self.m.basis,sel|dict(fixed_count=2))
        torch.testing.assert_close(got,batched,atol=.008,rtol=.004)
        ref,_=AttentionOps.evaluate(q,self.k,self.v,self.s,self.m.basis,mask.expand(-1,64,-1))
        torch.testing.assert_close(got,ref.bfloat16(),atol=.016,rtol=.004)
        sel=dict(mask=torch.ones(2,1,5,dtype=torch.bool),ids=torch.arange(5).expand(2,1,5),counts=torch.full((2,1),5))
        got=SelectedAttention.run(q,self.k,self.v,self.s,self.m.basis,sel)
        self.assertTrue(torch.isfinite(got).all())

    def test_calibration(self):
        from .calibration import fit_theta,calibrate_threshold
        f=torch.rand(2,12,5,4);target_theta=torch.tensor([.2,.3,.4,.1])
        labels=(f*target_theta).sum(-1)
        theta,fit=fit_theta(f,labels,steps=120)
        self.assertTrue((theta>=0).all())
        self.assertLess(fit['final_loss'],fit['initial_loss']*.05)
        cfg=Config()
        scores=[torch.rand(3,p) for p in (216,410)];own=[torch.arange(3) for _ in scores]
        row=calibrate_threshold(scores,own,cfg)
        self.assertLess(abs(row['actual_density']-.1),1/sum(x.numel() for x in scores))

    def test_calibration_column_scale_invariance(self):
        from .calibration import fit_theta
        f=torch.rand(2,20,7,4);truth=torch.tensor([.2,.3,.4,.1]);y=(f*truth).sum(-1)
        theta,info=fit_theta(f,y)
        scale=torch.tensor([1e-6,1e6,1e3,1.])
        scaled,info2=fit_theta(f*scale,y)
        torch.testing.assert_close((f*theta[:,None,None]).sum(-1),(f*scale*scaled[:,None,None]).sum(-1),atol=2e-5,rtol=2e-5)
        self.assertLess(info2['max_projected_gradient'],1e-5)

    def test_extreme_residual_label(self):
        from .ops import Summary
        q=torch.full((1,1,128),100.,dtype=torch.bfloat16)
        k=torch.zeros(1,1,64,128,dtype=torch.bfloat16);v=torch.zeros_like(k)
        s=Summary(torch.full((1,1,4,128),.5,dtype=torch.bfloat16),torch.ones(1,1,128,dtype=torch.bfloat16),torch.zeros(1,1,3,8,dtype=torch.bfloat16),torch.zeros(1,1,4,12,dtype=torch.bfloat16))
        y=AttentionOps.residual_labels(q,k,v,s,torch.zeros(1,128,8),torch.zeros(1,1,128),torch.full((1,1),torch.tensor(64.).log()))
        self.assertTrue(torch.isfinite(y).all())
        self.assertGreater(float(y.log()),500.)

    def test_summary_gradients(self):
        for name in ('pool_k','pool_v','pool_feature','basis'):getattr(self.m,name).requires_grad_()
        s=SummaryOps.build(self.m,self.k,self.v,cheap=True)
        out,_=AttentionOps.evaluate(self.q,self.k,self.v,s,self.m.basis,torch.zeros(2,4,5,dtype=torch.bool))
        out.square().mean().backward()
        for name in ('pool_k','pool_v','pool_feature','basis'):
            grad=getattr(self.m,name).grad
            self.assertTrue(torch.isfinite(grad).all() and grad.norm()>0,name)

if __name__=='__main__':unittest.main()
