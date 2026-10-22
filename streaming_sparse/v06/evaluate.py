"""Resumable paired evaluations with explicit baseline identity and raw mask evidence."""
import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import numpy as np
import torch
from .common import (HERE, DEFAULT_DATA, SnapshotData, atomic_json, query_slices,
                     source_identity, file_sha, protect_output)
from .model import load_router, from_base
from .metrics import error, summarize, trace_summary
from ..v05.primitives import statistics, mass_mask
from ..v03.metrics import partition_diagnostics
from ..evaluation.baselines import spatial_assignment


class FrozenCache:
    """Bounded CPU LRU; never retain a full dataset's QKV or autograd graphs."""
    def __init__(self, model, data, device, limit=8):
        self.model, self.data, self.device, self.limit = model, data, device, limit
        self.items = OrderedDict(); self.calls = 0

    @torch.no_grad()
    def get(self, version):
        if version not in self.items:
            events = self.data.load(version, self.device); first = events[0]
            state = self.model.precompute(first['k_hist'], first['v_hist'], head_id=first['head_id'], layer_id=first['layer_id'])
            state['partition_diagnostic'] = partition_diagnostics(state)
            state = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()
                     if k not in ('features', 'probabilities', 'raw_labels', 'usage_diagnostic')}
            cpu = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in event.items()} for event in events]
            self.items[version] = (cpu, state); self.calls += 1
            if len(self.items) > self.limit:
                self.items.popitem(last=False)
        cpu, state = self.items.pop(version); self.items[version] = (cpu, state)
        return ([{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in event.items()} for event in cpu],
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in state.items()})

    def stat(self, version, event, tile, selection, state):
        return statistics(event['q'][selection], event, state['assignment'])


def packed(mask):
    return np.packbits(mask.detach().cpu().numpy().astype(np.uint8), axis=-1, bitorder='little').tobytes().hex()


def compose(stage2, device='cuda', variants=False):
    """Extract separately selected correction weights; base identity must be exact."""
    directory = Path(stage2)
    names = ('C3_nocenter', 'C3_amplitude', 'C3_positive') if variants else ('C1', 'C2', 'C3')
    checkpoints = {name: directory / f'best_{name}.pt' for name in names}
    first, saved = load_router(checkpoints[names[0]], device)
    for name, path in checkpoints.items():
        other, value = load_router(path, device)
        if other.base_digest() != first.base_digest():
            raise ValueError('Cannot pair models with different frozen bases')
        if value['identity']['source_sha256'] != source_identity():
            raise ValueError('Source mismatch')
        first.branches[name].load_state_dict(other.branches[name].state_dict())
    first.freeze_base()
    return first, {k: file_sha(v) for k, v in checkpoints.items()}


@torch.no_grad()
def evaluate(model, data, selected, output, keeps=(.2,), checkpoint_sha=None, full=False, monitor=False):
    output = protect_output(output); output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    identity = dict(source_sha256=source_identity(), data_fingerprint=data.fingerprint(), split=data.split,
                    snapshots=sorted(selected), keeps=list(keeps), arms=list(model.arms), checkpoint_sha256=checkpoint_sha,
                    base_digest=model.base_digest(), full_diagnostics=full, monitor=monitor,
                    test_status='development Test informed prior design' if data.split == 'test' else data.split)
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('Evaluation identity mismatch')
    atomic_json(output / 'identity.json', identity)
    commits = output / 'snapshot_commits'; commits.mkdir(exist_ok=True)
    model.eval(); all_rows = []; partitions = []; diagnostics = []
    for index, version in enumerate(sorted(selected)):
        target_file = commits / f'{version}.json'
        if target_file.exists():
            saved = json.loads(target_file.read_text())
            if saved['identity'] != identity['source_sha256'] or len(saved['rows']) != 32 * len(keeps):
                raise ValueError('Corrupt snapshot commit')
        else:
            events = data.load(version, device); first = events[0]
            state = model.precompute(first['k_hist'], first['v_hist'], head_id=first['head_id'], layer_id=first['layer_id'])
            costs = state['hard'].sum(0); rows = []; ds = []
            spatial = spatial_assignment(first['token_positions'], 64) if full else None
            for event in events:
                for tile, selection in enumerate(query_slices(event, 16)):
                    q = event['q'][selection]; target = event['o_dense'][selection]
                    stat = statistics(q, event, state['assignment'])
                    traced = (full and tile == 0)
                    predictions = model.predict(q, state, trace=traced)
                    if traced:
                        for name, route in predictions.items():
                            ds.append(dict(model=name, trace=trace_summary(route, state),
                                correction_to_relevance_norm=float(route['scaled_correction'].norm() / route['relevance'].norm().clamp_min(1e-12))))
                    spatial_stat = statistics(q, event, spatial) if full else None
                    for keep in keeps:
                        routes = model.route(q, state, keep, predictions=predictions)
                        budget = math.floor(keep * len(state['hard']))
                        masks = {name: route['hard_mask'] for name, route in routes.items()}
                        masks['Mass'] = mass_mask(stat, costs, budget, state['block_ids'])
                        errors = {name: error(stat.output(mask), target) for name, mask in masks.items()}
                        for name, route in routes.items():
                            errors[name + '_soft'] = error(stat.output(route['soft_mask']), target)
                        retained = {name: (mask * costs).sum(-1).long().cpu().tolist() for name, mask in masks.items()}
                        if full:
                            sm = mass_mask(spatial_stat, spatial.sum(0), budget)
                            errors['SpatialMass'] = error(spatial_stat.output(sm), target)
                            retained['SpatialMass'] = (sm * spatial.sum(0)).sum(-1).long().cpu().tolist()
                        row = {k: event[k] for k in ('prompt_id', 'seed', 'layer_id', 'head_id', 'step_id')}
                        row.update(cache_version=version, tile_id=tile, keep=keep, history_tokens=len(state['hard']),
                                   budget=budget, partition_hash=state['partition_hash'], block_count=len(costs),
                                   errors=errors, retained=retained,
                                   mask_changes={name: (mask != masks['C0']).any(-1).cpu().tolist()
                                                 for name, mask in masks.items() if name in model.arms})
                        if full:
                            row.update(query_indices=event['query_indices'][selection].cpu().tolist(),
                                       masks_hex={name: packed(mask) for name, mask in masks.items()},
                                       mask_encoding='query-major16 rows, ceil(B/8) bytes per row, little-bitorder')
                        rows.append(row)
            saved = dict(identity=identity['source_sha256'], rows=rows,
                         partition=partition_diagnostics(state), diagnostics=ds)
            if full:
                saved.update(block_sizes=costs.long().cpu().tolist(), block_ids=state['block_ids'].cpu().tolist(),
                             token_block_labels=state['hard'].argmax(-1).cpu().tolist())
            atomic_json(target_file, saved)
        all_rows.extend(saved['rows']); partitions.append(saved['partition']); diagnostics.extend(saved['diagnostics'])
        atomic_json(output / 'progress.json', dict(completed=index + 1, total=len(selected), split=data.split))
    summary = dict(identity=identity, metrics=summarize(all_rows), capacity=model.capacity(),
                   partition_diagnostics=partitions, representation_diagnostics=diagnostics,
                   H_policy='One computation per snapshot, reused across four steps and all8 tiles')
    atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint'); source.add_argument('--stage2')
    parser.add_argument('--variants', action='store_true')
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--data', default=str(DEFAULT_DATA))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one CUDA GPU')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage2:
        model, sha = compose(args.stage2, 'cuda', args.variants)
    else:
        model, value = load_router(args.checkpoint, 'cuda'); sha = file_sha(args.checkpoint)
        if value['identity']['source_sha256'] != source_identity():
            raise ValueError('Checkpoint source mismatch')
    data = SnapshotData(args.data, args.split)
    result = evaluate(model, data, sorted(data.groups), args.output, (.3, .2, .1), sha, full=True)
    print(json.dumps({k: v for k, v in result['metrics'].items() if '/' not in k}), flush=True)


if __name__ == '__main__':
    main()
