"""One-snapshot numeric/backward correctness gate; no optimizer or Backbone."""
import argparse
import json
from pathlib import Path
import torch
from .model import RouterConfig, SparseRouter
from .data import SnapshotData
from .evaluate import source_identity, statistics
from .metrics import decoder_diagnostics, error_parts, gradient_gate, mask_change, partition_diagnostics
from .paths import protect_output
from ..attention.block_statistics import exact_attention
from ..capture.dataset_io import query_slices
from ..v02.evaluation_journal import atomic_json


def synthetic_events(device='cpu'):
    generator = torch.Generator().manual_seed(2026)
    kh, vh = [torch.randn(192, 128, generator=generator).to(device) for _ in range(2)]
    events = []
    for step in range(4):
        q = torch.randn(16, 128, generator=generator).to(device)
        kc, vc = [torch.randn(32, 128, generator=generator).to(device) for _ in range(2)]
        events.append(dict(k_hist=kh, v_hist=vh, k_current=kc, v_current=vc, q=q,
            o_dense=exact_attention(q, torch.cat((kh, kc)), torch.cat((vh, vc))),
            token_positions=torch.zeros(192, 3, device=device), head_id=0, layer_id=4,
            query_indices=torch.arange(16, device=device), query_tile=16, step_id=step))
    return events


def check(model, events):
    first = events[0]
    state = model.precompute(first['k_hist'], first['v_hist'], first['token_positions'], first['head_id'], first['layer_id'])
    state['features'].retain_grad()
    state['probabilities'].retain_grad()
    cached_h = state['h'].detach().clone()
    losses, masks, diagnostics, scores = [], [], [], []
    for event in events:
        selection = query_slices(event, 16)[0]
        q, target = event['q'][selection], event['o_dense'][selection]
        stats = statistics(q, event, state['assignment'])
        route = model.route(q, state, .3)
        torch.testing.assert_close(stats.dense(), target.float(), atol=5e-5, rtol=5e-5)
        selected = state['hard']@route['hard_mask'] > 0
        direct = exact_attention(q, torch.cat((event['k_hist'][selected], event['k_current'])),
                                 torch.cat((event['v_hist'][selected], event['v_current'])))
        torch.testing.assert_close(stats.output(route['mask']), direct, atol=5e-5, rtol=5e-5)
        if int(selected.sum()) > route['budget']:
            raise AssertionError('Budget exceeded')
        se, en = error_parts(stats.output(route['mask']), target)
        losses.append(se/(en+1e-8))
        masks.append(selected)
        scores.append(route['scores'].detach())
        with torch.no_grad():
            diagnostics.append(decoder_diagnostics(route, stats, target, state['hard']))
    loss = torch.stack(losses).mean()
    pg, = torch.autograd.grad(loss, state['probabilities'], retain_graph=True)
    fg, = torch.autograd.grad(state['probabilities'], state['features'], grad_outputs=pg, retain_graph=True)
    state['cluster_feature_path_norm'] = float(fg.norm())
    loss.backward()
    gradients = gradient_gate(model, state)
    torch.testing.assert_close(cached_h, state['h'].detach(), atol=0, rtol=0)
    score_changes = [float((a-b).abs().max()) for a, b in zip(scores, scores[1:])]
    if not any(value > 1e-6 for value in score_changes):
        raise AssertionError('Different Q produced identical scores')
    return dict(passed=True, dense_correctness=True, raw_kv_gather_correctness=True,
                attention_loss=float(loss.detach()), gradients=gradients, precompute_calls=1,
                diffusion_steps=len(events), score_changes=score_changes, h_cache_unchanged=True,
                step_mask_jaccard=mask_change(masks), partition=partition_diagnostics(state),
                decoder=diagnostics, parameter_counts=model.parameter_counts())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic', action='store_true')
    mode.add_argument('--data')
    p.add_argument('--snapshot')
    p.add_argument('--initial-subset', action='store_true', help='Real first-capture subset, before full Stage A collection')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.synthetic:
        events = synthetic_events(args.device)
    else:
        data = SnapshotData(args.data, 'train', require_complete=not args.initial_subset)
        version = args.snapshot or sorted(data.groups)[0]
        events = data.load(version, args.device)
    report = dict(version='0.3', real_qkv=not args.synthetic, device=args.device,
                  full_data_coverage=not args.synthetic and not args.initial_subset,
                  source_sha256=source_identity(), optimizer_updates=0, fusions={})
    for fusion in ('concat', 'film'):
        torch.manual_seed(0)
        model = SparseRouter(RouterConfig(fusion=fusion, target_size=16 if args.synthetic else 64)).to(args.device)
        report['fusions'][fusion] = check(model, events)
    target = protect_output(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
