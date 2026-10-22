"""Real-QKV full36-pair correctness plus discarded-model optimizer smoke checks."""
import argparse
import json
import os
import torch
from .common import DEFAULT_DATA, SnapshotData, atomic_json, protect_output, source_identity, query_slices
from .model import Router
from .train import update
from ..v05.primitives import statistics
from ..attention.block_statistics import exact_attention


def check(model, events):
    first = events[0]; calls = []
    handle = model.frontend.register_forward_hook(lambda *args: calls.append(1))
    state = model.precompute(first['k_hist'], first['v_hist'], head_id=first['head_id'], layer_id=first['layer_id'])
    h = state['h'].detach().clone(); losses = []; signs = []
    for event in events:
        selection = query_slices(event, 16)[0]; q = event['q'][selection]; target = event['o_dense'][selection].float()
        stat = statistics(q, event, state['assignment'])
        torch.testing.assert_close(stat.dense(), target, atol=5e-5, rtol=5e-5)
        routes = model.route(q, state, .2, trace=True)
        changed = q.clone(); changed[0] += torch.linspace(-1, 1, 128, device=q.device)
        after = model.predict(changed, state)
        for name, route in routes.items():
            if route['scores'].shape != (16, len(state['h'])):
                raise AssertionError('Query axis pooled')
            torch.testing.assert_close(route['scores'][1:], after[name]['scores'][1:], atol=0, rtol=0)
            if float((route['scores'][0] - after[name]['scores'][0]).abs().max()) <= 1e-8:
                raise AssertionError('Query inactive')
            if route['fusion'].data_ptr() != routes['C0']['fusion'].data_ptr():
                raise AssertionError('Different query fusion tensors')
            torch.testing.assert_close(route['relevance'], routes['C0']['relevance'], atol=0, rtol=0)
            retained = (route['hard_mask'] * state['hard'].sum(0)).sum(-1)
            if bool((retained > route['budget']).any()):
                raise AssertionError('Budget exceeded')
            for i in (0, 15):
                tokens = state['hard'] @ route['hard_mask'][i] > 0
                direct = exact_attention(q[i:i + 1], torch.cat((event['k_hist'][tokens], event['k_current'])),
                                         torch.cat((event['v_hist'][tokens], event['v_current'])))
                torch.testing.assert_close(direct, stat.output(route['hard_mask'])[i:i + 1], atol=5e-5, rtol=5e-5)
            losses.append((stat.output(route['mask']) - target).square().sum() / (target.square().sum() + 1e-8))
        for layer in routes['C3']['layers']:
            r = layer['relation']
            if r.dtype != torch.float32 or float(r.diagonal(dim1=-1, dim2=-2).abs().sum()) != 0:
                raise AssertionError('Signed FP32 or diagonal failure')
            signs.append(dict(positive=bool((r > 1e-4).any()), negative=bool((r < -1e-4).any())))
    loss = torch.stack(losses).sum()
    pg, = torch.autograd.grad(loss, state['probabilities'], retain_graph=True)
    if not torch.isfinite(pg).all() or float(pg.norm()) == 0:
        raise AssertionError('No clustering gradient')
    loss.backward(); norms = {}
    for name, module in [('frontend', model.frontend), ('fusion', model.query_fusion), ('relevance', model.relevance), *model.branches.items()]:
        gs = [p.grad for p in module.parameters() if p.grad is not None]
        if not gs or any(not torch.isfinite(g).all() for g in gs):
            raise AssertionError('Missing/nonfinite gradient: ' + name)
        norms[name] = sum(float(g.square().sum()) for g in gs) ** .5
        if norms[name] <= 0:
            raise AssertionError('Zero gradient: ' + name)
    handle.remove()
    torch.testing.assert_close(state['h'].detach(), h, atol=0, rtol=0)
    if calls != [1] or not all(s['positive'] and s['negative'] for s in signs):
        raise AssertionError('Cache or signed geometry failure')
    return dict(passed=True, partition_hash=state['partition_hash'], H_calls=1, steps=4,
                query_shape=list(routes['C0']['scores'].shape), dense_and_raw_gather=True,
                identical_baseline_scores=True, shared_fusion=True, query_independence=True,
                gradient_norms=norms, clustering_gradient=float(pg.norm()), signed_relations=signs)


def smoke(data, version):
    torch.manual_seed(909); c0 = Router().cuda()
    optimizer = torch.optim.AdamW(c0.parameters(), lr=1e-4)
    before = c0.base_digest()
    one = update(c0, data, [(version, 0)], optimizer, ['C0'], trace=True)
    if before == c0.base_digest():
        raise AssertionError('C0 optimizer did not update')
    model = Router(arms=('C1', 'C2', 'C3'), correction_seed=100909).cuda()
    own = model.state_dict(); own.update(dict(c0.base_items())); model.load_state_dict(own); model.freeze_base()
    before = model.base_digest()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    two = update(model, data, [(version, 0)], optimizer, model.arms, trace=True)
    if before != model.base_digest():
        raise AssertionError('Frozen-base optimizer drift')
    return dict(passed=True, discarded_model_updates=2, c0_update=one, frozen_corrections_update=two,
                frozen_base_bitwise_unchanged=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--single', action='store_true'); args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    data = SnapshotData(DEFAULT_DATA, 'train'); chosen = [min(v) for _, v in sorted(data.strata.items())]
    if args.single:
        chosen = chosen[:1]
    reports = []
    for i, version in enumerate(chosen):
        torch.manual_seed(0); model = Router(arms=('C1', 'C2', 'C3')).cuda(); events = data.load(version, 'cuda')
        reports.append(dict(cache_version=version, layer=events[0]['layer_id'], head=events[0]['head_id'], **check(model, events)))
        print(json.dumps(dict(completed=i + 1, total=len(chosen), layer=events[0]['layer_id'], head=events[0]['head_id'])), flush=True)
        del events, model
    optimizer_smoke = smoke(data, chosen[0])
    target = protect_output(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, dict(passed=True, full_head_coverage=len(reports) == 36, real_qkv=True,
        source_sha256=source_identity(), data_fingerprint=data.fingerprint(), gpu_name=torch.cuda.get_device_name(0),
        torch=str(torch.__version__), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        checks=reports, optimizer_smoke=optimizer_smoke))


if __name__ == '__main__':
    main()
