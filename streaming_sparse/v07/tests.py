"""Exact mathematics, no-leak deployable path, independent frozen controls."""
import copy
import inspect
import math
import unittest
import torch
from .exact import (teacher,current_stats,pages,singleton,set_gain,geometry,MaskOutput,page_costs,
                    TeacherStats,retained_residual)
from .sketch import Sketch
from .router import Router
from ..v03.sanity import synthetic_events
from ..attention.block_statistics import exact_attention
from ..v05.primitives import select,BudgetGate


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def example(self):
        torch.manual_seed(7);event=synthetic_events()[0]
        model=Router(dict(prototypes=4,page_size=16),('R3-local','R4'))
        state=model.sketch(event['k_hist'],event['v_hist'],0)
        return model,event,state

    def test_exact_outputs_and_identity_all_budgets(self):
        model,e,state=self.example();q=e['q'];stats=teacher(q,e,16);g=stats.geometry()
        torch.testing.assert_close(stats.dense(),e['o_dense'],atol=1e-6,rtol=1e-5)
        costs=state['costs']
        for rho in (0,.1,.2,.3,1):
            mask=select(torch.randn_like(stats.logz),costs,math.floor(rho*len(e['k_hist'])))
            torch.testing.assert_close(stats.output(mask)-stats.dense(),
                -((1-mask)[...,None]*g['d']).sum(1)/(g['current_mass']+(mask*g['mass']).sum(-1))[:,None],atol=1e-6,rtol=2e-5)
            for i in (0,15):
                tokens=mask[i].repeat_interleave(16).bool()
                direct=exact_attention(q[i:i+1],torch.cat((e['k_hist'][tokens],e['k_current'])),torch.cat((e['v_hist'][tokens],e['v_current'])))
                torch.testing.assert_close(stats.output(mask)[i:i+1],direct,atol=1e-6,rtol=1e-5)

    def test_logsumexp_current_merge(self):
        _,e,_=self.example();s=teacher(e['q'],e,64)
        Lhist=torch.logsumexp(s.logz,-1);oh=(s.logz-Lhist[:,None]).exp()[...,None]*s.mu;oh=oh.sum(1)
        Lall=torch.logaddexp(Lhist,s.current_logz)
        merged=(Lhist-Lall).exp()[:,None]*oh+(s.current_logz-Lall).exp()[:,None]*s.current_output
        torch.testing.assert_close(merged,e['o_dense'],atol=1e-6,rtol=1e-5)

    def test_p1_mean_mathematics_multiple_q_examples(self):
        for seed,n in ((0,97),(1,192),(2,140)):
            torch.manual_seed(seed);k=torch.randn(n,128);v=torch.randn(n,128);s=Sketch(1,page_size=64);state=s(k,v,0)
            kk,vv,valid=pages(k,v,64);counts=valid.sum(-1);means=(vv*valid[...,None]).sum(1)/counts[:,None]
            for q in (torch.zeros(3,128),torch.randn(5,128)):
                current=(torch.zeros(len(q)),torch.zeros(len(q),128))
                p=s.estimate(q,state,current)
                kmean=(kk*valid[...,None]).sum(1)/counts[:,None]
                L=q@kmean.T/math.sqrt(128)+counts.float().log()[None]
                torch.testing.assert_close(p['logz'],L,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(p['mu'],means[None].expand(len(q),-1,-1),atol=1e-6,rtol=1e-5)
                if q.count_nonzero()==0:torch.testing.assert_close(p['logz'].exp(),counts.float()[None].expand_as(L),atol=2e-5,rtol=1e-5)

    def test_gain_matches_true_finite_add_remove_errors(self):
        _,e,state=self.example();t=teacher(e['q'],e,16);g=t.geometry();m=select(torch.randn_like(t.logz),state['costs'],38)
        gain=set_gain(g,m);E=(t.output(m)-t.dense()).square().sum(-1)
        torch.testing.assert_close(gain['error'],E,atol=1e-6,rtol=1e-5)
        for b in range(m.shape[1]):
            after=m.clone();after[:,b]=1-m[:,b]
            Ea=(t.output(after)-t.dense()).square().sum(-1)
            expected=torch.where(m[:,b]>.5,Ea-E,E-Ea)
            torch.testing.assert_close(gain['gain'][:,b],expected,atol=1e-6,rtol=1e-4)

    def test_small_retained_mass_cancellation_regression(self):
        torch.manual_seed(0)
        L=torch.randn(2,128)*2+2;mu=torch.randn(2,128,128)*.2+2
        Lc=torch.full((2,),-3.);oc=torch.randn(2,128)
        t=TeacherStats(L,mu,Lc,oc);mask=torch.zeros_like(L);g=t.geometry()
        actual=t.output(mask)-t.dense();D,z=retained_residual(g,mask)
        stable=(-D/z[:,None]).float()
        torch.testing.assert_close(actual,stable,atol=5e-5,rtol=5e-5)
        # Explicitly prove this fixture exposes the original failure mode.
        total=torch.logsumexp(torch.cat((L,Lc[:,None]),-1),-1)
        mass=(L-total[:,None]).exp();cm=(Lc-total).exp()
        dense=(mass[...,None]*mu).sum(1)+cm[:,None]*oc
        naive=-(mass[...,None]*(mu-dense[:,None])).sum(1)/cm[:,None]
        self.assertGreater(float((naive-actual).abs().max()),1e-3)
        gain=set_gain(g,mask)
        torch.testing.assert_close(gain['error'],actual.square().sum(-1),atol=5e-5,rtol=5e-5)

    def test_stable_gain_preserves_initial_mask_surrogate_derivative(self):
        torch.manual_seed(5)
        L=torch.randn(2,6,dtype=torch.double);mu=torch.randn(2,6,128,dtype=torch.double)
        Lc=torch.zeros(2,dtype=torch.double);oc=torch.randn(2,128,dtype=torch.double)
        g=geometry(L,mu,Lc,oc)
        m=torch.tensor([[1.,0.,1.,0.,0.,0.],[0.,1.,0.,0.,1.,0.]],dtype=torch.double,requires_grad=True)
        stable=set_gain(g,m)['gain']
        D=((1-m)[...,None]*g['d']).sum(1);z=g['current_mass']+(m*g['mass']).sum(-1)
        E=(D/z[:,None]).square().sum(-1)
        Ea=(D[:,None]-g['d']).square().sum(-1)/(z[:,None]+g['mass']).square()
        Er=(D[:,None]+g['d']).square().sum(-1)/(z[:,None]-g['mass']).clamp_min(1e-12).square()
        original=torch.where(m.detach()>.5,Er-E[:,None],E[:,None]-Ea)
        torch.testing.assert_close(stable,original,atol=1e-9,rtol=1e-9)
        direction=torch.randn_like(m)
        a,=torch.autograd.grad((stable*direction).sum(),m,retain_graph=True)
        b,=torch.autograd.grad((original*direction).sum(),m)
        torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-8)

    def test_current_value_can_flip_optimal_mask(self):
        d=128;kh=torch.zeros(3,d);vh=torch.zeros(3,d);vh[:,0]=torch.tensor([-1.,0.,1.]);q=torch.zeros(1,d)
        sk=Sketch(1,page_size=1);state=sk(kh,vh,0);masks=[]
        for value in (-2.,2.):
            current=torch.zeros(1,d);current[:,0]=value
            pred=sk.estimate(q,state,(torch.zeros(1),current))
            masks.append(select(singleton(pred),state['costs'],1))
        self.assertFalse(torch.equal(*masks))

    def test_perquery_independence_and_final_partial_page(self):
        model,e,state=self.example();current=current_stats(e['q'],e['k_current'],e['v_current'])
        before=model.route(e['q'],state,current);changed=e['q'].clone();changed[2]*=-4
        after=model.route(changed,state,current_stats(changed,e['k_current'],e['v_current']))
        for n,r in before.items():
            keep=torch.arange(16)!=2
            torch.testing.assert_close(r['scores'][keep],after[n]['scores'][keep],atol=0,rtol=0)
            self.assertEqual(r['hard_mask'].shape,(16,len(state['costs'])))
            self.assertTrue(((r['hard_mask']*state['costs']).sum(-1)<=38).all())
        self.assertEqual(page_costs(97,64,'cpu').tolist(),[64,33])
        mask=select(torch.tensor([[2.,1.]]),page_costs(97,64,'cpu'),33)
        self.assertEqual(mask.tolist(),[[0.,1.]])

    def test_budget_derivatives_and_exact_mask_derivative(self):
        torch.manual_seed(1)
        scores=torch.randn(2,4,dtype=torch.double,requires_grad=True);costs=torch.tensor([2.,3.,4.,1.],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda s,c:BudgetGate.apply(s,c,5,.2),(scores,costs),eps=1e-5,atol=2e-4))
        L=torch.randn(2,4,dtype=torch.double);mu=torch.randn(2,4,128,dtype=torch.double);Lc=torch.randn(2,dtype=torch.double);oc=torch.randn(2,128,dtype=torch.double)
        mask=torch.full((2,4),.4,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda m:MaskOutput.apply(m,L,mu,Lc,oc),(mask,),eps=1e-5,atol=1e-5))

    def test_extreme_logits_empty_prototypes_and_masks(self):
        L=torch.tensor([[1000.,-1000.]]);mu=torch.randn(1,2,128);Lc=torch.tensor([-2000.]);oc=torch.randn(1,128)
        for values in ([[0.,0.]],[[0.,1.]],[[1.,0.]],[[1.,1.]]):
            m=torch.tensor(values,requires_grad=True);out=MaskOutput.apply(m,L,mu,Lc,oc)
            self.assertTrue(torch.isfinite(out).all());out.square().sum().backward();self.assertTrue(torch.isfinite(m.grad).all())
            if values==[[0.,0.]]:torch.testing.assert_close(out,oc,atol=0,rtol=0)
        sk=Sketch(4,page_size=64)
        with torch.no_grad():sk.weight.zero_();sk.bias[:,0]=1e4
        state=sk(torch.randn(97,128),torch.randn(97,128),0)
        self.assertGreater(int((~state['valid']).sum()),0)
        pred=sk.estimate(torch.randn(3,128),state,(torch.zeros(3),torch.zeros(3,128)))
        self.assertTrue(torch.isfinite(pred['dense']).all())

    def test_no_teacher_input_and_value_changes_teacher_only(self):
        model,e,state=self.example();sig=inspect.signature(model.route)
        self.assertEqual(set(sig.parameters),{'q','state','current','keep','history_only','shuffle'})
        current=current_stats(e['q'],e['k_current'],e['v_current']);before=model.route(e['q'],state,current)
        fake=dict(e,v_hist=e['v_hist']*50,o_dense=e['o_dense']*50);teacher(e['q'],fake,16)
        after=model.route(e['q'],state,current)
        for n in before:torch.testing.assert_close(before[n]['scores'],after[n]['scores'],atol=0,rtol=0)

    def test_key_only_assignment_mass_does_not_read_values(self):
        _,e,_=self.example();s=Sketch(4,key_only=True,page_size=16)
        a=s(e['k_hist'],e['v_hist'],0);b=s(e['k_hist'],e['v_hist']*20,0)
        q=e['q'];ca=current_stats(q,e['k_current'],e['v_current']);cb=current_stats(q,e['k_current'],e['v_current']*-20)
        pa=s.estimate(q,a,ca);pb=s.estimate(q,b,cb)
        torch.testing.assert_close(pa['mass'],pb['mass'],atol=0,rtol=0)
        self.assertGreater(float((pa['mu']-pb['mu']).abs().max()),0)

    def test_frozen_base_and_capacity_and_gradient(self):
        model,e,state=self.example()
        self.assertEqual(sum(p.numel() for p in model.corrections['R3-local'].parameters()),sum(p.numel() for p in model.corrections['R4'].parameters()))
        model.freeze_base();digest=model.base_digest();state=model.sketch(e['k_hist'],e['v_hist'],0)
        routes=model.route(e['q'],state,current_stats(e['q'],e['k_current'],e['v_current']))
        true=teacher(e['q'],e,16);loss=sum((true.output(routes[n]['mask'])-e['o_dense']).square().mean() for n in model.arms)
        loss.backward()
        for n,m in model.corrections.items():
            gs=[p.grad for p in m.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gs));self.assertGreater(sum(float(g.square().sum()) for g in gs),0)
        self.assertTrue(all(p.grad is None for m in (model.sketch,model.base) for p in m.parameters()))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4);opt.step()
        self.assertEqual(digest,model.base_digest())

    def test_sketch_and_joint_initial_ste_gradients(self):
        model,e,state=self.example();current=current_stats(e['q'],e['k_current'],e['v_current']);true=teacher(e['q'],e,16)
        route=model.route(e['q'],state,current)['R4'];torch.testing.assert_close(route['mask'].detach(),route['hard_mask'],atol=0,rtol=0)
        loss=(true.output(route['mask'])-e['o_dense']).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(model.sketch.weight.grad).all());self.assertGreater(float(model.sketch.weight.grad.norm()),0)
        self.assertGreater(float(model.base.gamma.grad.norm()),0)


if __name__=='__main__':unittest.main()
