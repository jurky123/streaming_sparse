"""Interventions on independent queries, budget derivatives and signed geometry."""
import unittest
import torch
from .model import Router, RouterConfig
from .primitives import statistics, select, BudgetGate
from ..router.budget_selection import soft_budget_relaxation
from ..attention.block_statistics import exact_attention
from ..v03.sanity import synthetic_events


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def make(self):
        torch.manual_seed(4)
        model=Router(RouterConfig(target_size=16,bf16=False))
        events=synthetic_events()
        state=model.precompute(events[0]['k_hist'],events[0]['v_hist'])
        return model,events,state

    def test_statistics_all_and_per_query_gather(self):
        model,events,state=self.make();event=events[0];q=event['q']
        stat=statistics(q,event,state['assignment'])
        torch.testing.assert_close(stat.dense(),event['o_dense'],atol=1e-6,rtol=1e-5)
        hard=model.route(q,state)['R3']['hard_mask']
        outputs=[]
        for index,mask in enumerate(hard):
            tokens=state['hard']@mask>0
            outputs.append(exact_attention(q[index:index+1],torch.cat((event['k_hist'][tokens],event['k_current'])),
                                           torch.cat((event['v_hist'][tokens],event['v_current']))))
        torch.testing.assert_close(stat.output(hard),torch.cat(outputs),atol=1e-6,rtol=1e-5)

    def test_query_axis_is_independent(self):
        model,events,state=self.make();q=events[0]['q']
        before=model.route(q,state)
        changed=q.clone();changed[3]=changed[3]*-4
        after=model.route(changed,state)
        for name in model.arms:
            self.assertEqual(before[name]['scores'].shape,(16,len(state['h'])))
            untouched=torch.arange(16)!=3
            torch.testing.assert_close(before[name]['scores'][untouched],after[name]['scores'][untouched],rtol=0,atol=0)
            self.assertGreater(float((before[name]['scores'][3]-after[name]['scores'][3]).abs().max()),1e-6)

    def test_budget_and_permutation_with_ties(self):
        scores=torch.tensor([[3.,2.,2.,1.],[0.,0.,0.,0.]])
        costs=torch.tensor([4.,3.,0.,2.]);ids=torch.tensor([9,1,6,3])
        mask=select(scores,costs,5,ids);p=torch.tensor([2,0,3,1])
        other=select(scores[:,p],costs[p],5,ids[p])
        torch.testing.assert_close(mask[:,p],other,rtol=0,atol=0)
        self.assertTrue(bool(((mask*costs).sum(-1)<=5).all()))
        self.assertEqual(float(mask[:,2].sum()),0)

    def test_vectorized_gate_matches_original_and_cost_gradient(self):
        torch.manual_seed(1)
        scores=torch.randn(3,5,dtype=torch.double,requires_grad=True)
        costs=torch.tensor([2.,3.,4.,1.,2.],dtype=torch.double,requires_grad=True)
        batch=BudgetGate.apply(scores,costs,5,.2)
        old=torch.stack([soft_budget_relaxation(s,costs,5,.2) for s in scores])
        torch.testing.assert_close(batch,old,rtol=1e-10,atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(lambda s,c:BudgetGate.apply(s,c,5,.2),(scores,costs),eps=1e-5,atol=2e-4))

    def test_signed_relations_empty_self_and_zero_correction(self):
        model,events,state=self.make()
        # Add a truly empty block; it must not change any valid output.
        state=dict(state);b=len(state['h'])
        for key in ('h','z','z_normalized'):
            state[key]=torch.cat((state[key],torch.zeros_like(state[key][:1])),0)
        for key in ('hard','assignment'):
            state[key]=torch.cat((state[key],torch.zeros(len(state[key]),1)),1)
        state['empty']=torch.cat((state['empty'],torch.tensor([True])))
        state['block_ids']=torch.arange(b+1)
        routes=model.route(events[0]['q'],state,trace=True)
        for layer in routes['R3']['layers']:
            relation=layer['relation']
            self.assertGreater(float(relation.max()),0)
            self.assertLess(float(relation.min()),0)
            self.assertEqual(float(relation.diagonal(dim1=-2,dim2=-1).abs().sum()),0)
            self.assertEqual(float(relation[:,-1].abs().sum()+relation[:,:,-1].abs().sum()),0)
            self.assertLessEqual(float(relation.abs().max()),1)
        zero=model.route(events[0]['q'],state,intervention='zero')['R3']
        torch.testing.assert_close(zero['correction'],torch.zeros_like(zero['correction']),rtol=0,atol=0)
        torch.testing.assert_close(zero['scores'][:,:-1],zero['relevance'][:,:-1],rtol=0,atol=0)

    def test_score_mask_block_permutation(self):
        model,events,state=self.make();q=events[0]['q']
        before=model.route(q,state);p=torch.randperm(len(state['h']))
        other=dict(state)
        for key in ('h','z','z_normalized','empty','block_ids'):other[key]=state[key][p]
        for key in ('hard','assignment','probabilities'):other[key]=state[key][:,p]
        after=model.route(q,other)
        for name in model.arms:
            torch.testing.assert_close(before[name]['scores'][:,p],after[name]['scores'],atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(before[name]['hard_mask'][:,p],after[name]['hard_mask'],rtol=0,atol=0)

    def test_all_branches_and_cluster_gradients_and_cache(self):
        model,events,state=self.make();h=state['h'].detach().clone();losses=[]
        for event in events:
            stat=statistics(event['q'],event,state['assignment'])
            for route in model.route(event['q'],state).values():
                losses.append((stat.output(route['mask'])-event['o_dense']).square().sum()/event['o_dense'].square().sum())
        loss=torch.stack(losses).mean()
        grad,=torch.autograd.grad(loss,state['probabilities'],retain_graph=True)
        self.assertGreater(float(grad.norm()),0)
        loss.backward()
        for module in [model.frontend,model.query_fusion,*model.branches.values()]:
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.square().sum()) for g in grads),0)
        signed=model.branches['R3']
        for block in signed.interaction:
            self.assertGreater(sum(float(p.grad.square().sum()) for p in block.u.parameters() if p.grad is not None),0)
        self.assertGreater(float(signed.correction_scale.grad.abs()),0)
        torch.testing.assert_close(h,state['h'],rtol=0,atol=0)

    def test_all_arms_share_exact_query_fusion(self):
        model,events,state=self.make()
        count=[]
        handle=model.query_fusion.register_forward_hook(lambda *args:count.append(1))
        routes=model.route(events[0]['q'],state,trace=True)
        handle.remove()
        self.assertEqual(count,[1])
        for name in model.arms:
            self.assertEqual(routes[name]['fusion'].data_ptr(),routes['R2']['fusion'].data_ptr())
            self.assertEqual(routes[name]['query_embedding'].data_ptr(),routes['R2']['query_embedding'].data_ptr())

    def test_update_uses_pooled_energy_not_mean_tile_ratios(self):
        from .train import update
        model,events,state=self.make()
        events[1]['q']=events[1]['q']*4
        events[1]['o_dense']=exact_attention(events[1]['q'],torch.cat((events[1]['k_hist'],events[1]['k_current'])),
                                            torch.cat((events[1]['v_hist'],events[1]['v_current'])))
        class Data:
            def load(self,version,device):return events
        # CPU cache exercises the same fixed-prefix path as phase1 without CUDA.
        class Cache:
            def get(self,version):return events,state
            def stat(self,version,event,tile,selection,state):return statistics(event['q'][selection],event,state['assignment'])
        expected={name:[] for name in model.arms};energy=0.
        for event in events:
            stat=statistics(event['q'],event,state['assignment']);energy+=float(event['o_dense'].square().sum())
            for name,route in model.route(event['q'],state).items():
                expected[name].append(float((stat.output(route['mask'])-event['o_dense']).square().sum()))
        optimizer=torch.optim.AdamW(model.parameters(),lr=0,weight_decay=0)
        result=update(model,Data(),[('fake',0)],optimizer,Cache())
        for name in model.arms:
            self.assertAlmostEqual(result['attention_loss'][name],sum(expected[name])/energy,places=6)
        self.assertGreater(abs(result['attention_loss']['R2']-result['tile_mean_relative_error']['R2']),1e-5)

    def test_evaluation_pairing_and_final_metric(self):
        from .metrics import group_summary
        row=dict(cache_version='x',step_id=0,tile_id=0,history_tokens=10,budget=2,
                 retained={'R2':[2,1],'R3':[1,2]},errors={
                     'R2':dict(sse=4.,energy=10.,error=.4,per_query_error=[.1,.8]),
                     'R3':dict(sse=3.,energy=10.,error=.3,per_query_error=[.2,.5])})
        summary=group_summary([row])
        self.assertAlmostEqual(summary['paired']['query_win_rate_R3'],.5)
        self.assertAlmostEqual(summary['R3']['global_attention_error'],3/(10+1e-8))
        self.assertAlmostEqual(summary['R3']['query_relative_error']['mean'],.35)
        self.assertNotEqual(summary['R3']['query_relative_error']['mean'],summary['R3']['global_attention_error'])


if __name__=='__main__':unittest.main()
