"""Synthetic software tests only. No restored QKV, Backbone, or research runs."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch
from .model import RouterConfig, SparseRouter, load_router
from .data import SnapshotData, MissingQKVError
from .evaluate import evaluate, statistics
from .metrics import error_parts, gradient_gate, summarize
from .train import run
from .merge import merge
from .workflow import plan
from ..attention.block_statistics import exact_attention
from ..capture.dataset_io import CaptureWriter, atomic_save


def tensors(seed=0, history=96):
    gen = torch.Generator().manual_seed(seed)
    kh, vh = [torch.randn(history, 128, generator=gen) for _ in range(2)]
    kc, vc = [torch.randn(24, 128, generator=gen) for _ in range(2)]
    q = torch.randn(32, 128, generator=gen)
    return dict(k_hist=kh, v_hist=vh, k_current=kc, v_current=vc, q=q,
                o_dense=exact_attention(q, torch.cat((kh, kc)), torch.cat((vh, vc))),
                token_positions=torch.stack((torch.arange(history)//32, torch.arange(history)%8, torch.arange(history)//8), -1))


def make_dataset(root):
    protocol = json.loads((Path(__file__).parent/'configs/protocol.json').read_text())
    writer = CaptureWriter(root)
    for index, split in enumerate(('train', 'val', 'test')):
        sample = tensors(index)
        prompt = protocol['splits'][split][0]
        history = {k: sample[k] for k in ('k_hist', 'v_hist', 'token_positions')}
        for step in range(4):
            event = tensors(index*10+step+20)
            event.update({k: sample[k] for k in ('k_hist', 'v_hist', 'token_positions')})
            event['o_dense'] = exact_attention(event['q'], torch.cat((sample['k_hist'], event['k_current'])),
                                              torch.cat((sample['v_hist'], event['v_current'])))
            event.update(query_indices=torch.arange(32), query_tile=16)
            metadata = dict(prompt_id=prompt, seed=0, layer_id=4, head_id=0, step_id=step,
                            chunk_id=1, history_tokens=96, query_tile=16, captured_queries=32,
                            query_tile_starts=[0, 16], total_queries=32)
            writer.write(history, event, metadata)
    (Path(root)/'splits.json').write_text(json.dumps(protocol['splits']))


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, fusion='concat', **kwargs):
        torch.manual_seed(42)
        return SparseRouter(RouterConfig(fusion=fusion, target_size=8, max_blocks=16, **kwargs))

    def test_dense_gather_budget_and_gradients_for_both_fusions(self):
        sample = tensors()
        for fusion in ('concat', 'film'):
            model = self.model(fusion)
            state = model.precompute(sample['k_hist'], sample['v_hist'], sample['token_positions'], 0, 4)
            state['features'].retain_grad()
            state['probabilities'].retain_grad()
            torch.testing.assert_close(state['assignment'], state['hard'], rtol=0, atol=0)
            self.assertTrue(bool((state['hard'].sum(0) > 0).all()))
            q = sample['q'][:16]
            stats = statistics(q, sample, state['assignment'])
            torch.testing.assert_close(stats.dense(), sample['o_dense'][:16], atol=5e-5, rtol=5e-5)
            for keep in (0., .1, .2, .3, 1.):
                route = model.route(q, state, keep)
                kept = state['hard']@route['hard_mask'] > 0
                self.assertLessEqual(int(kept.sum()), route['budget'])
                self.assertAlmostEqual(float((route['soft_mask']*state['hard'].sum(0)).sum()), route['budget'], places=3)
                direct = exact_attention(q, torch.cat((sample['k_hist'][kept], sample['k_current'])),
                                         torch.cat((sample['v_hist'][kept], sample['v_current'])))
                torch.testing.assert_close(stats.output(route['mask']).detach(), direct, atol=5e-5, rtol=5e-5)
            route = model.route(q, state, .3)
            se, en = error_parts(stats.output(route['mask']), sample['o_dense'][:16])
            loss = se/(en+1e-8)
            pg, = torch.autograd.grad(loss, state['probabilities'], retain_graph=True)
            fg, = torch.autograd.grad(state['probabilities'], state['features'], grad_outputs=pg, retain_graph=True)
            state['cluster_feature_path_norm'] = float(fg.norm())
            loss.backward()
            norms = gradient_gate(model, state)
            self.assertTrue(all(v > 0 for v in norms.values()), norms)
            for name, parameter in model.named_parameters():
                if 'delta_' in name and parameter.grad is not None:
                    self.assertEqual(float(parameter.grad[1:].abs().sum()), 0., name)

    def test_cache_position_independence_and_block_permutation(self):
        sample = tensors()
        for fusion in ('concat', 'film'):
            model = self.model(fusion).eval()
            with torch.no_grad():
                state = model.precompute(sample['k_hist'], sample['v_hist'], sample['token_positions'])
                shifted = model.precompute(sample['k_hist'], sample['v_hist'], sample['token_positions']+10000)
                torch.testing.assert_close(state['h'], shifted['h'], atol=0, rtol=0)
                cached = state['h'].clone()
                with patch.object(model, 'precompute', side_effect=AssertionError('H recomputed in route')):
                    a = model.route(sample['q'][:16], state, .3)
                    b = model.route(sample['q'][16:], state, .3)
                torch.testing.assert_close(state['h'], cached, atol=0, rtol=0)
                self.assertGreater(float((a['scores']-b['scores']).abs().max()), 1e-6)
                perm = torch.randperm(len(state['h']))
                rearranged = dict(state)
                for key in ('h', 'empty'):
                    rearranged[key] = state[key][perm]
                for key in ('hard', 'assignment', 'probabilities'):
                    rearranged[key] = state[key][:, perm]
                c = model.route(sample['q'][:16], rearranged, .3)
                torch.testing.assert_close(c['scores'], a['scores'][perm], atol=2e-5, rtol=2e-5)

    def test_parameter_capacity_pair_specialization_and_serialization(self):
        sample = tensors()
        with tempfile.TemporaryDirectory() as folder:
            for fusion in ('concat', 'film'):
                model = self.model(fusion)
                counts = model.parameter_counts()
                self.assertLess(counts['pair_specific'], counts['shared'])
                a = model.precompute(sample['k_hist'], sample['v_hist'], head_id=0, layer_id=4)
                b = model.precompute(sample['k_hist'], sample['v_hist'], head_id=1, layer_id=4)
                self.assertGreater(float((a['features']-b['features']).detach().abs().max()), 1e-6)
                path = Path(folder)/f'{fusion}.pt'
                atomic_save(model.checkpoint(), path)
                restored = load_router(path)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
            shared = self.model(conditioned=False)
            self.assertEqual(shared.parameter_counts()['pair_specific'], 0)
            with self.assertRaises(ValueError):
                shared.pair_index(99, 4)

    def test_prompt_head_macro_metric(self):
        rows = []
        # Unequal head energies: macro=(1.0+0.1)/2=.55; global=11/101.
        for head, se, en in [(0, 1., 1.), (1, 10., 100.)]:
            rows.append(dict(prompt_id='p', layer_id=4, head_id=head, squared_error=se,
                             dense_energy=en, error=se/en, actual_keep=.2, budget_utilization=1.))
        report = summarize(rows, bootstrap=20)
        self.assertAlmostEqual(report['macro_prompt_head_relative_error'], .55, places=6)
        self.assertAlmostEqual(report['global_relative_error'], 11/101, places=6)

    def test_missing_tensors_and_conflicting_cells_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            make_dataset(folder)
            root = Path(folder)
            rows = (root/'events.jsonl').read_text().splitlines()
            (root/'events.jsonl').write_text('\n'.join(rows+[rows[0]])+'\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                SnapshotData(root, 'train', heads=(0,), layers=(4,), require_complete=False)
            (root/'events.jsonl').write_text('\n'.join(rows)+'\n')
            data = SnapshotData(root, 'train', heads=(0,), layers=(4,), require_complete=False)
            (root/data.dataset.records[0]['history']).unlink()
            with self.assertRaises(MissingQKVError):
                SnapshotData(root, 'train', heads=(0,), layers=(4,), require_complete=False)

    def test_evaluation_journal_recovery_and_b3_partition_pairing(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            make_dataset(root/'data')
            data = SnapshotData(root/'data', 'val', heads=(0,), layers=(4,), require_complete=False)
            model = self.model()
            original = data.load
            # Fail after writing tile rows but before the snapshot is committed.
            from ..v02.evaluation_journal import EvaluationJournal
            commit = EvaluationJournal.commit
            def interrupt(journal, done):
                if done == 1:
                    raise RuntimeError('simulated interruption')
                return commit(journal, done)
            with patch.object(EvaluationJournal, 'commit', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    evaluate(model, data, root/'eval', bootstrap=20)
            result = evaluate(model, data, root/'eval', bootstrap=20)
            again = evaluate(model, data, root/'eval', bootstrap=20)
            self.assertEqual(result, again)
            rows = [json.loads(x) for x in (root/'eval/per_tile.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 4*2*3*5)
            self.assertEqual({r['method'] for r in rows}, {'B0', 'B1', 'B2', 'B3', 'B4'})
            self.assertIn('B4/keep=0.2', result['comparisons'])
            self.assertEqual({r['partition_source'] for r in rows if r['method'] in ('B3', 'B4')}, {'concat'})
            with torch.no_grad():
                next(model.parameters()).add_(.1)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                evaluate(model, data, root/'eval', bootstrap=20)

    def test_training_resume_during_validation_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            make_dataset(root/'data')
            config = dict(steps=2, seed=0, heads=[0], layers=[4], require_complete=False,
                router=dict(heads=[0], layers=[4], target_size=8, max_blocks=16),
                validate_every=1, save_every=1, log_every=1, loss='attention_output_only')
            uninterrupted = run(config, root/'data', root/'reference', device='cpu')
            with patch('streaming_sparse.v03.train.evaluate', side_effect=RuntimeError('interrupted validation')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted validation'):
                    run(config, root/'data', root/'resumed', device='cpu')
            resumed = run(config, root/'data', root/'resumed', device='cpu', resume=root/'resumed/latest.pt')
            for name, value in uninterrupted.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
            self.assertEqual(len((root/'resumed/validation_curve.jsonl').read_text().splitlines()), 2)
            checkpoint = torch.load(root/'resumed/best.pt', weights_only=True)
            self.assertEqual(checkpoint['selection_split'], 'val')

    def test_existing_adapter_reuses_h_and_retains_current_kv(self):
        from ..integration.self_forcing_adapter import SelfForcingAdapter
        sample = tensors()
        model = self.model().eval()
        adapter = SelfForcingAdapter(object(), router=model, layers=[4], heads=[0], keep_ratio=0.)
        def attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        adapter.original = attention
        context = dict(history_tokens=96, layer_id=4, positions=sample['token_positions'],
                       history_signature=('snapshot', 0), prompt_id='p', chunk_id=1, step_id=0)
        q = sample['q'][None, :, None]
        k = torch.cat((sample['k_hist'], sample['k_current']))[None, :, None]
        v = torch.cat((sample['v_hist'], sample['v_current']))[None, :, None]
        with patch.object(model, 'precompute', wraps=model.precompute) as precompute:
            output = adapter._sparse(context, q, k, v)
            adapter._sparse({**context, 'step_id': 1}, q+1, k, v)
            self.assertEqual(precompute.call_count, 1)
        torch.testing.assert_close(output, attention(q, k[:, 96:], v[:, 96:]), atol=5e-5, rtol=5e-5)

    def test_content_addressed_merge_rejects_conflicts_before_writing(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_dataset(root/'a')
            shutil.copytree(root/'a', root/'b')
            result = merge([root/'a', root/'b'], root/'merged', heads=(0,), require_complete=False)
            self.assertEqual(result['events'], 12)
            self.assertEqual(result['duplicate_identical_cells'], 12)
            restored = SnapshotData(root/'merged', 'train', heads=(0,), layers=(4,), require_complete=False)
            self.assertEqual(len(restored.load(next(iter(restored.groups)), 'cpu')), 4)
            row = json.loads((root/'b/events.jsonl').read_text().splitlines()[0])
            path = root/'b'/row['event']
            value = torch.load(path, weights_only=True)
            value['q'] = value['q']+1
            atomic_save(value, path)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                merge([root/'a', root/'b'], root/'conflict', heads=(0,), require_complete=False)
            self.assertFalse((root/'conflict').exists())

    def test_plans_and_output_protection(self):
        from .paths import protect_output
        full = plan('full', '/data', '/runs')
        self.assertEqual(len([x for x in full['jobs'] if x['kind'] == 'train']), 6)
        pilot = plan('pilot', '/data', '/runs')
        self.assertEqual(len([x for x in pilot['jobs'] if x['kind'] == 'train']), 2)
        self.assertTrue(all(x.get('split') != 'test' for x in pilot['jobs']))
        with self.assertRaises(ValueError):
            protect_output(Path(__file__).parents[1]/'v02/runs/accidental_v03')

    def test_architecture_gate_rejects_synthetic_and_test_evidence(self):
        from .gate import architecture_gate
        from .evaluate import source_identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summaries, training, overfit = [], [], []
            for fusion in ('concat', 'film'):
                directory = root/fusion
                directory.mkdir()
                method = 'B4' if fusion == 'concat' else 'B5'
                config = dict(router=dict(fusion=fusion), stage='pilot')
                prov = dict(source_sha256=source_identity(), validation_data_fingerprint='data')
                (directory/'config.json').write_text(json.dumps(config))
                (directory/'provenance.json').write_text(json.dumps(prov))
                row = dict(attention_loss=1., first_update_gradient_gate={'feature': 1.}, diagnostics=[])
                (directory/'metrics.jsonl').write_text((json.dumps(row)+'\n')*1500)
                summary = dict(fusion=fusion, split='val', complete_data_protocol=True, heads=[0, 3, 6, 9],
                               snapshots=288, source_sha256=source_identity(), data_fingerprint='data', training_identity=prov,
                               metrics={f'{method}/keep=0.2': {'macro_prompt_head_relative_error': .8},
                                        'B3/keep=0.2': {'macro_prompt_head_relative_error': 1.}})
                sp = directory/'summary.json'
                sp.write_text(json.dumps(summary))
                op = directory/'overfit.json'
                op.write_text(json.dumps(dict(fusion=fusion, passed=True, real_qkv=True, source_sha256=source_identity())))
                summaries.append(sp); training.append(directory); overfit.append(op)
            self.assertTrue(architecture_gate(summaries, training, overfit)['capture_allowed'])
            x = json.loads(summaries[0].read_text())
            x['split'] = 'test'
            summaries[0].write_text(json.dumps(x))
            self.assertFalse(architecture_gate(summaries, training, overfit)['capture_allowed'])
            x['split'], x['complete_data_protocol'] = 'val', False
            summaries[0].write_text(json.dumps(x))
            self.assertFalse(architecture_gate(summaries, training, overfit)['capture_allowed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
