"""Meaningful intervention checks before running real multi-snapshot fits."""
import unittest
import torch
from .model import RepairRouter
from ..model import RouterConfig
from ..sanity import synthetic_events
from ..evaluate import statistics
from ..metrics import error_parts


class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.events = synthetic_events()
        self.base = RepairRouter(RouterConfig(target_size=16))
        self.initial = {k:v.clone() for k,v in self.base.state_dict().items()}
        with torch.no_grad():
            e = self.events[0]
            self.frozen = {k:v.detach() if isinstance(v, torch.Tensor) else v for k,v in
                           self.base.precompute(e['k_hist'], e['v_hist']).items()}

    def model(self, arm):
        model = RepairRouter(RouterConfig(target_size=16))
        model.load_state_dict(self.initial)
        return model.configure(arm)

    def test_initial_outputs_match_all_interventions(self):
        e = self.events[0]
        reference = self.base.route(e['q'], self.frozen, .2)
        for arm in 'ABCD':
            m = self.model(arm)
            s = m.fixed_precompute(e, self.frozen) if arm in 'AB' else m.precompute(e['k_hist'], e['v_hist'])
            route = m.route(e['q'], s, .2)
            torch.testing.assert_close(reference['scores'], route['scores'], rtol=2e-5, atol=1e-6)
            torch.testing.assert_close(reference['hard_mask'], route['hard_mask'], rtol=0, atol=0)

    def test_fixed_partitions_survive_feature_updates(self):
        m = self.model('B')
        e = self.events[0]
        frozen = self.frozen['hard'].clone()
        state = m.fixed_precompute(e, self.frozen)
        loss = m.route(e['q'], state, .2)['scores'].square().sum()
        loss.backward()
        self.assertGreater(sum(float(p.grad.norm()) for p in m.features.parameters() if p.grad is not None), 0)
        self.assertFalse(state['assignment'].requires_grad)
        with torch.no_grad():
            for p in m.features.parameters():
                p.add_(torch.randn_like(p)*.2)
        after = m.fixed_precompute(e, self.frozen)
        torch.testing.assert_close(frozen, after['hard'], rtol=0, atol=0)
        self.assertFalse(torch.allclose(state['features'], after['features']))

    def test_expected_gradient_paths_and_fixed_F(self):
        for arm in 'ABCD':
            m = self.model(arm)
            e = self.events[0]
            state = m.fixed_precompute(e, self.frozen) if arm in 'AB' else m.precompute(e['k_hist'], e['v_hist'])
            route = m.route(e['q'], state, .2)
            se, en = error_parts(statistics(e['q'], e, state['assignment']).output(route['mask']), e['o_dense'])
            if arm in 'CD':
                gradient, = torch.autograd.grad(se/en, state['probabilities'], retain_graph=True)
                self.assertGreater(float(gradient.norm()), 0)
            (se/en).backward()
            grads = [p.grad for p in m.features.parameters()]
            self.assertEqual(all(g is None for g in grads), arm == 'A')
            if arm != 'A':
                self.assertGreater(sum(float(g.norm()) for g in grads if g is not None), 0)
            self.assertGreater(sum(float(p.grad.norm()) for p in m.decoder.parameters() if p.grad is not None), 0)


if __name__ == '__main__':
    unittest.main()
