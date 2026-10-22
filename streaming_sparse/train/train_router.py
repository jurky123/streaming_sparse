"""Offline router-only training; H is reused only within one optimizer step."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import random
import time
from pathlib import Path
import torch
from ..capture.dataset_io import AttentionDataset, atomic_save, query_slices
from ..router.model import SparseRouter, RouterConfig
from ..attention.block_statistics import differentiable_block_stats
from ..attention.reconstruction_loss import normalized_attention_mse


def to_device(sample, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}


def gradient_norms(model):
    modules = {'grouping': model.features.group, 'content': model.features.content,
               'aggregator': model.aggregator, 'encoder': model.encoder, 'decoder': model.decoder}
    return {name: float(sum((p.grad.detach().float().square().sum() for p in module.parameters()
                             if p.grad is not None), torch.tensor(0., device=next(module.parameters()).device)).sqrt())
            for name, module in modules.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    seed = cfg.get('seed', 0)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = AttentionDataset(args.data, 'train')
    if not len(dataset):
        raise ValueError('Empty train split')
    groups = defaultdict(list)
    for i, record in enumerate(dataset.records):
        groups[record['cache_version']].append(i)
    groups = list(groups.values())
    model = SparseRouter(RouterConfig(**cfg.get('router', {}))).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 1e-4), weight_decay=cfg.get('weight_decay', 0.01))
    begin = 0
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=True, map_location=args.device)
        if ckpt['config'] != asdict(model.config):
            raise ValueError('Resume router configuration differs')
        model.load_state_dict(ckpt['router'])
        optimizer.load_state_dict(ckpt['optimizer'])
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if args.device.startswith('cuda') and 'cuda_rng' in ckpt:
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt['cuda_rng']])
        random.setstate(ckpt['python_rng'])
        begin = ckpt['step'] + 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.jsonl').exists() and not args.resume:
        raise FileExistsError('Training output already contains metrics; use a new output or --resume')
    (output / 'training_config.json').write_text(json.dumps(cfg, indent=2))
    if not args.resume:
        atomic_save(model.checkpoint(), output / 'initial_router.pt')
    for step in range(begin, cfg.get('steps', 2000)):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        indices = random.choice(groups)
        chosen = random.sample(indices, min(len(indices), cfg.get('events_per_step', 4)))
        samples = [to_device(dataset[i], args.device) for i in chosen]
        first = samples[0]
        state = model.precompute(first['k_hist'], first['v_hist'], first['token_positions'],
                                 first['head_id'], first['layer_id'])
        loss, soft_errors, retained = 0, [], []
        keep = random.choice(cfg.get('keep_ratios', [0.3, 0.2]))
        for sample in samples:
            tile = model.config.query_tile
            # Saved query indices identify tile boundaries; no interpolation of nonadjacent Q.
            selection = random.choice(query_slices(sample, tile))
            q = sample['q'][selection]
            route = model.route(q, state, keep)
            stats = differentiable_block_stats(q, sample['k_hist'], sample['v_hist'], state['assignment'],
                                               sample['k_current'], sample['v_current'])
            target = sample['o_dense'][selection].float()
            loss = loss + normalized_attention_mse(stats.output(route['mask']), target) / len(samples)
            with torch.no_grad():
                soft_errors.append(float(normalized_attention_mse(stats.output(route['soft_mask']), target)))
                retained.append(float((state['hard'].sum(0) * route['hard_mask']).sum()) / len(sample['k_hist']))
        attention_loss = loss
        loss = loss + cfg.get('collapse_weight', 1e-3) * state['regularizer']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        gradients = gradient_norms(model)
        if any(not math_isfinite(x) for x in gradients.values()):
            raise FloatingPointError(f'Invalid gradients: {gradients}')
        if cfg.get('require_gradients', True) and step == begin:
            required = ['content', 'aggregator', 'encoder', 'decoder']
            if model.config.grouping == 'learned':
                required.append('grouping')
            if any(gradients[x] <= 0 for x in required):
                raise RuntimeError(f'Zero module gradient: {gradients}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip', 1.0), error_if_nonfinite=True)
        optimizer.step()
        sizes = state['hard'].sum(0).detach()
        record = dict(step=step, hard_loss=float(attention_loss.detach()), soft_loss=sum(soft_errors)/len(soft_errors),
                      actual_keep=sum(retained)/len(retained), requested_keep=keep, blocks=len(sizes),
                      block_min=float(sizes.min()), block_max=float(sizes.max()), gradients=gradients,
                      elapsed_ms=(time.perf_counter()-tick)*1000)
        record['block_sizes'] = sizes.to(device='cpu',dtype=torch.long).tolist()
        record['block_median'] = float(sizes.median())
        record['block_p90'] = float(sizes.quantile(.9))
        record['soft_assignment_entropy'] = float(-(state['probabilities'].detach() *
            state['probabilities'].detach().clamp_min(1e-8).log()).sum(-1).mean())
        record['hard_soft_gap'] = record['hard_loss'] - record['soft_loss']
        with (output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if step % cfg.get('log_every', 10) == 0:
            print(json.dumps(record), flush=True)
        if (step+1) % cfg.get('save_every', 100) == 0 or step == cfg.get('steps', 2000)-1:
            ckpt = {**model.checkpoint(), 'optimizer': optimizer.state_dict(), 'step': step,
                    'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}
            if args.device.startswith('cuda'):
                ckpt['cuda_rng'] = torch.cuda.get_rng_state_all()
            atomic_save(ckpt, output / 'latest.pt')
            atomic_save(model.checkpoint(), output / 'router.pt')


def math_isfinite(value):
    import math
    return math.isfinite(value)


if __name__ == '__main__':
    main()
