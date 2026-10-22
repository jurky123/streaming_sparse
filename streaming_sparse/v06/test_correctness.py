"""Causal controls, immutable base, independent optimization and exact diagnostics."""
import copy
import unittest
import torch
from .model import Router, RouterConfig
from .common import tensor_digest
from .train import update
from .diagnostics import omitted_identity, approx_greedy
from .metrics import group_summary
from ..v05.primitives import statistics
from ..v03.sanity import synthetic_events


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, arms=('C1', 'C2', 'C3')):
        torch.manual_seed(6)
        model = Router(RouterConfig(target_size=16, bf16=False), arms)
        events = synthetic_events(); state = model.precompute(events[0]['k_hist'], events[0]['v_hist'])
        return model, events, state

    def test_c0_genuinely_has_no_correction_and_matches_initial_base(self):
        model, events, state = self.make(())
        self.assertEqual(len(model.branches), 0)
        other, _, _ = self.make()
        self.assertEqual(model.base_digest(), other.base_digest())
        a = model.predict(events[0]['q'], state)['C0']['scores']
        b = other.predict(events[0]['q'], state)['C0']['scores']
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_c1_is_local_and_capacity_matched(self):
        model, events, state = self.make(); x, _ = model.query_fusion(events[0]['q'], state['h'], state['empty'], state['pair'])
        before, _ = model.branches['C1'](x, state)
        changed = x.clone(); changed[:, 2] += torch.linspace(-10, 10, 128)
        after, _ = model.branches['C1'](changed, state)
        untouched = torch.arange(len(state['h'])) != 2
        torch.testing.assert_close(before[:, untouched], after[:, untouched], atol=0, rtol=0)
        self.assertGreater(float((before[:, 2] - after[:, 2]).abs().max()), 1e-6)
        self.assertEqual(model.capacity()['correction']['C1'], model.capacity()['correction']['C2'])

    def test_query_independence_and_block_permutation(self):
        model, events, state = self.make(); q = events[0]['q']; routes = model.route(q, state)
        altered = q.clone(); altered[0] *= -3
        changed = model.route(altered, state)
        p = torch.randperm(len(state['h'])); other = dict(state)
        for key in ('h', 'z', 'z_normalized', 'empty', 'block_ids'):
            other[key] = state[key][p]
        for key in ('hard', 'assignment'):
            other[key] = state[key][:, p]
        permuted = model.route(q, other)
        for name, route in routes.items():
            self.assertEqual(route['scores'].shape, (16, len(state['h'])))
            torch.testing.assert_close(route['scores'][1:], changed[name]['scores'][1:], atol=0, rtol=0)
            torch.testing.assert_close(route['scores'][:, p], permuted[name]['scores'], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(route['hard_mask'][:, p], permuted[name]['hard_mask'], atol=0, rtol=0)
            self.assertTrue(bool(((route['hard_mask'] * state['hard'].sum(0)).sum(-1) <= route['budget']).all()))

    def test_frozen_base_and_independent_optimization(self):
        model, events, _ = self.make(); model.freeze_base(); before = model.base_digest()
        state = model.precompute(events[0]['k_hist'], events[0]['v_hist'])
        class Cache:
            def get(self, version):
                return events, state
        # Frozen state has no autograd graph. Disjoint branch sum + separate clipping
        # must equal a truly separate single-arm optimizer update.
        single = copy.deepcopy(model); single.branches = torch.nn.ModuleDict({'C2': single.branches['C2']}); single.arms = ('C2',)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        opt_single = torch.optim.AdamW([p for p in single.parameters() if p.requires_grad], lr=1e-4)
        update(model, None, [('fake', 0)], opt, model.arms, Cache())
        update(single, None, [('fake', 0)], opt_single, single.arms, Cache())
        self.assertEqual(before, model.base_digest())
        self.assertEqual(tensor_digest(model.branches['C2'].state_dict().items()), tensor_digest(single.branches['C2'].state_dict().items()))
        self.assertTrue(all(p.grad is None for module in (model.frontend, model.query_fusion, model.relevance) for p in module.parameters()))

    def test_c0_and_corrected_gradients(self):
        for arms in ((), ('C2',), ('C3',)):
            model, events, state = self.make(arms)
            stat = statistics(events[0]['q'], events[0], state['assignment'])
            routes = model.route(events[0]['q'], state); name = arms[0] if arms else 'C0'
            loss = (stat.output(routes[name]['mask']) - events[0]['o_dense']).square().mean()
            grad, = torch.autograd.grad(loss, state['probabilities'], retain_graph=True)
            self.assertTrue(bool(torch.isfinite(grad).all())); self.assertGreater(float(grad.norm()), 0)
            loss.backward()
            for module in (model.frontend, model.query_fusion, model.relevance, *model.branches.values()):
                gs = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertGreater(sum(float(g.square().sum()) for g in gs), 0)
                self.assertTrue(all(torch.isfinite(g).all() for g in gs))

    def test_signed_variants_are_actual_training_paths(self):
        model, events, state = self.make(('C3', 'C3_nocenter', 'C3_amplitude', 'C3_positive'))
        routes = model.route(events[0]['q'], state, trace=True)
        for name in model.arms:
            relation = routes[name]['layers'][0]['relation']
            self.assertEqual(float(relation.diagonal(dim1=-1, dim2=-2).abs().sum()), 0)
            if name == 'C3_positive':
                self.assertGreaterEqual(float(relation.min()), 0)
            elif name == 'C3':
                self.assertLess(float(relation.min()), 0); self.assertGreater(float(relation.max()), 0)
            stat = statistics(events[0]['q'], events[0], state['assignment'])
            loss = (stat.output(routes[name]['mask']) - events[0]['o_dense']).square().mean()
            grad, = torch.autograd.grad(loss, model.branches[name].interaction[0].u.linears[0].base.weight, retain_graph=True)
            self.assertTrue(bool(torch.isfinite(grad).all())); self.assertGreater(float(grad.norm()), 0)
        torch.testing.assert_close(model.branches['C3'].interaction[0].u.linears[0].base.weight,
                                   model.branches['C3_positive'].interaction[0].u.linears[0].base.weight, atol=0, rtol=0)

    def test_exact_interaction_identity_and_greedy_budget(self):
        model, events, state = self.make(); event = events[0]
        stat = statistics(event['q'], event, state['assignment']); dense = stat.dense()
        torch.testing.assert_close(dense, event['o_dense'], atol=1e-6, rtol=1e-5)
        mask = model.route(event['q'], state)['C3']['hard_mask']
        torch.testing.assert_close(omitted_identity(stat, dense, mask), stat.output(mask) - dense, atol=1e-6, rtol=1e-5)
        single = statistics(event['q'][:1], event, state['assignment'])
        costs = state['hard'].sum(0); budget = 38
        greedy, info = approx_greedy(single, event['o_dense'][:1], costs, budget)
        self.assertLessEqual(int((greedy * costs).sum()), budget)
        empty = torch.zeros_like(greedy)
        self.assertLessEqual(float((single.output(greedy) - event['o_dense'][:1]).square().sum()),
                             float((single.output(empty) - event['o_dense'][:1]).square().sum()) + 1e-7)

    def test_pooled_objective_and_changed_mask_outcomes(self):
        row = dict(cache_version='x', step_id=0, tile_id=0, history_tokens=10, budget=2,
                   retained={'C0':[2]*16, 'C2':[1]*16}, mask_changes={'C2':[True]*8+[False]*8},
                   errors={'C0':dict(sse=4., energy=10., error=.4, per_query_error=[.1,.8]*8),
                           'C2':dict(sse=3., energy=10., error=.3, per_query_error=[.2,.5]*8)})
        value = group_summary([row])
        self.assertAlmostEqual(value['C2']['global_attention_error'], 3/(10+1e-8))
        self.assertEqual(value['paired']['C2_vs_C0']['query_win_rate'], .5)
        self.assertEqual(value['correction_effect']['C2']['changed_fraction'], .5)
        self.assertEqual(value['correction_effect']['C2']['improved_given_change'], .5)


if __name__ == '__main__':
    unittest.main()
