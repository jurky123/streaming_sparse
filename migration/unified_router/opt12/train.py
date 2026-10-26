"""8-worker layer-parallel joint tuning with complete, append-only loss history."""
import argparse
import gc
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import torch
from ..common import config, load_raw, save, atomic_json, update, paired_seed, event_key
from ..opt11.data import dense
from .model import JointRouter
from .precision import clip_grad_norm


def append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(dict(time=time.time(), **record), allow_nan=False) + '\n')
        f.flush()


class Events:
    """Raw data are mmap'ed on CPU; only the active event is copied to CUDA."""
    def __init__(self, m, layer):
        self.items = []
        for row in m['rows']:
            query = torch.load(Path(m['group_cache']) / event_key(row) / f'layer{layer:02d}.pt',
                               map_location='cpu', weights_only=True, mmap=True)
            assert query['sample_id'] == row['sample_id']
            assert query['indices'] == row['group_query_indices']
            raw = load_raw(m['parent_experiment'], row, layer, device='cpu')
            raw.pop('q')
            self.items.append(dict(row=row, q=query['q'], raw=raw, truth=None))
        self.train = [x for x in self.items if x['row']['split'] == 'train']
        self.val = [x for x in self.items if x['row']['split'] == 'val']
        assert len(self.train) == 192 and len(self.val) == 96

    def gpu(self, item):
        q = item['q'].cuda()
        raw = {k: v.cuda() for k, v in item['raw'].items()}
        if item['truth'] is None:
            with torch.no_grad():
                item['truth'] = dense(q, raw).float().cpu()
        return q, raw, item['truth'].cuda()


@torch.no_grad()
def evaluate(model, events):
    rows = []
    for item in events.val:
        q, raw, truth = events.gpu(item)
        y, mask, d = model.deployed(q, raw, item['row']['step'])
        sse = (y - truth).square().sum()
        energy = truth.square().sum()
        keep = (mask * d['costs']).sum(-1).float().mean() / (d['history_tokens'] + 4680)
        rows.append(dict(sample_id=item['row']['sample_id'], nmse=float(sse / energy),
                         sse=float(sse), energy=float(energy), keep=float(keep)))
    return dict(nmse=sum(x['nmse'] for x in rows) / len(rows),
                keep=sum(x['keep'] for x in rows) / len(rows),
                energy_weighted_nmse=sum(x['sse'] for x in rows) / sum(x['energy'] for x in rows),
                worst_event_nmse=max(x['nmse'] for x in rows), rows=rows)


def checkpoint(model, layer, variant, step, value, cap):
    return dict(backward_revision='fp32_all_surrogate_backward_v3', model=model.state_dict(), head=model.head.state_dict(),
                parameters={k: v.detach() for k, v in model.dictionary().items()},
                layer=layer, variant=variant, updates=step, best_value=value,
                best_metric='actual_deployment_attention_nmse', max_best_keep=cap)


def run(root, m, layers, rank):
    status = root / f'training/rank{rank}.json'
    for layer in layers:
        folder = root / f'training/layer{layer:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'completed.json').exists():
            continue
        paired_seed(12000 + layer)
        update(status, status='loading', layer=layer, updates=0, total=m['max_updates'])
        events = Events(m, layer)
        models = {v: JointRouter(m['parent_experiment'], m['parent_opt11'], layer, v, m['monitor_beta']).cuda()
                  for v in m['variants']}
        opts = {v: torch.optim.AdamW([
                    dict(params=list(model.routing.parameters()), lr=m['learning_rate_router'], name='router'),
                    dict(params=list(model.head.parameters()), lr=m['learning_rate_head'], name='head')], weight_decay=0.)
                for v, model in models.items()}
        rng = random.Random(12000 + layer)
        best, curves, caps, initial, significant, stale = {}, {}, {}, {}, {}, {}
        start = 0
        recovery = folder / 'recovery.pt'
        if recovery.exists():
            state = torch.load(recovery, map_location='cpu', weights_only=True)
            start = state['updates']
            for v in m['variants']:
                models[v].load_state_dict(state['models'][v])
                opts[v].load_state_dict(state['optimizers'][v])
            best, curves, caps = state['best'], state['curves'], state['caps']
            initial, significant, stale = state['initial'], state['significant'], state['stale']
            rng.setstate(state['rng'])
        else:
            for v, model in models.items():
                update(status, status='initial_validation', layer=layer, variant=v, updates=0, total=m['max_updates'])
                result = evaluate(model, events)
                parent_validation = json.loads((Path(m['parent_opt11']) / f'strong_training/layer{layer:02d}/{v}/best_validation.json').read_text())
                assert abs(result['nmse'] - parent_validation['nmse']) < 1e-6, (layer, v, result['nmse'], parent_validation['nmse'])
                initial[v] = best[v] = significant[v] = result['nmse']
                stale[v] = 0
                # Some inherited layers start above16%; do not allow them to expand.
                caps[v] = max(m['max_best_keep'], result['keep'])
                curves[v] = [dict(updates=0, **{k: x for k, x in result.items() if k != 'rows'})]
                for name in ['initial', 'best']:
                    save(folder / v / (name + '.pt'), checkpoint(model, layer, v, 0, best[v], caps[v]))
                    atomic_json(folder / v / (name + '_validation.json'), dict(updates=0, **result))
                atomic_json(folder / v / 'curve.json', curves[v])
                print(f'layer={layer} {v} initial NMSE={best[v]:.8f} keep={result["keep"]:.5f}', flush=True)
        append(folder / 'resume_history.jsonl', dict(rank=rank, start_updates=start, revision='fp32_all_surrogate_backward_v3'))
        last_step = start
        reason = 'max_updates'
        for step in range(start + 1, m['max_updates'] + 1):
            if (root / 'STOP').exists():
                reason = 'stop_file'
                break
            t0 = time.perf_counter()
            chosen = rng.sample(events.train, m['batch_events'])
            ratio = min(1., step / m['warmup_updates']) * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / m['max_updates'])))
            metrics = {v: dict(actual_nmse=0., keep=0., entropy=0., budget_loss=0.) for v in models}
            for v, opt in opts.items():
                opt.zero_grad(set_to_none=True)
                opt.param_groups[0]['lr'] = m['learning_rate_router'] * ratio
                opt.param_groups[1]['lr'] = m['learning_rate_head'] * ratio
            for item in chosen:
                q, raw, truth = events.gpu(item)
                energy = truth.square().sum().clamp_min(1e-20)
                for v, model in models.items():
                    actual, keep, entropy = model(q, raw, item['row']['step'], m['gate_temperature'], m['action_temperature'])
                    nmse = (actual - truth).square().sum() / energy
                    budget = m['budget_penalty'] * (keep - m['target_keep']).clamp_min(0).square()
                    loss = (nmse + budget) / len(chosen)
                    assert torch.isfinite(loss), (layer, v, step)
                    loss.backward()
                    for k, x in dict(actual_nmse=nmse, keep=keep, entropy=entropy, budget_loss=budget).items():
                        metrics[v][k] += float(x.detach()) / len(chosen)
                    del actual, keep, entropy, loss, nmse, budget
                del q, raw, truth
            for v, model in models.items():
                try:
                    rg = clip_grad_norm(model.routing.parameters(), 1.)
                    hg = clip_grad_norm(model.head.parameters(), 1.)
                except RuntimeError:
                    bad = {n: int((~torch.isfinite(p.grad)).sum()) for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()}
                    update(status, status='failed', layer=layer, updates=last_step, failed_update=step, variant=v, bad_gradients=bad, event_ids=[x['row']['sample_id'] for x in chosen])
                    raise
                metrics[v].update(router_grad_norm=float(rg), head_grad_norm=float(hg),
                                  lr_router=m['learning_rate_router'] * ratio,
                                  lr_head=m['learning_rate_head'] * ratio)
            # Validate both variants before mutating either optimizer/model.
            for opt in opts.values():
                opt.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            for v in models:
                append(folder / v / 'train.jsonl', dict(updates=step, revision='fp32_all_surrogate_backward_v3', **metrics[v], update_seconds=elapsed,
                       event_ids=[x['row']['sample_id'] for x in chosen]))
            last_step = step
            update(status, status='training', layer=layer, updates=step, total=m['max_updates'],
                   losses=metrics, best=best, initial=initial, update_seconds=elapsed,
                   peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            if step == start + 1 or step % 25 == 0:
                print(f'layer={layer} step={step} seconds={elapsed:.2f} metrics={metrics}', flush=True)
            if step % m['validate_every'] == 0 or step == m['max_updates']:
                for v, model in models.items():
                    update(status, status='validating', layer=layer, variant=v, updates=step,
                           total=m['max_updates'], best=best, initial=initial)
                    result = evaluate(model, events)
                    eligible = result['keep'] <= caps[v]
                    record = dict(updates=step, eligible=eligible, **{k: x for k, x in result.items() if k != 'rows'})
                    curves[v].append(record)
                    atomic_json(folder / v / 'curve.json', curves[v])
                    atomic_json(folder / v / f'validation/update{step:06d}.json', dict(updates=step, eligible=eligible, **result))
                    if eligible and result['nmse'] < best[v]:
                        best[v] = result['nmse']
                        save(folder / v / 'best.pt', checkpoint(model, layer, v, step, best[v], caps[v]))
                        atomic_json(folder / v / 'best_validation.json', dict(updates=step, eligible=True, **result))
                    if eligible and result['nmse'] < significant[v] * (1 - m['min_relative_improvement']):
                        significant[v] = result['nmse']
                        stale[v] = 0
                    else:
                        stale[v] += 1
                    print(f'validation layer={layer} {v} step={step} NMSE={result["nmse"]:.8f} keep={result["keep"]:.5f} best={best[v]:.8f} eligible={eligible}', flush=True)
            plateau = step >= m['min_updates'] and all(stale[v] >= m['patience_validations'] for v in models)
            if step % m['checkpoint_every'] == 0 or plateau or step == m['max_updates']:
                save(recovery, dict(updates=step, models={v: x.state_dict() for v, x in models.items()},
                     optimizers={v: x.state_dict() for v, x in opts.items()}, curves=curves, best=best,
                     initial=initial, caps=caps, significant=significant, stale=stale, rng=rng.getstate()))
            if plateau:
                reason = 'validation_plateau_after_min_updates'
                break
        for v, model in models.items():
            save(folder / v / 'last.pt', checkpoint(model, layer, v, last_step, best[v], caps[v]))
        if reason == 'stop_file':
            # Save a resumable state at the exact last completed step, including RNG.
            save(recovery, dict(updates=last_step, models={v: x.state_dict() for v, x in models.items()},
                 optimizers={v: x.state_dict() for v, x in opts.items()}, curves=curves, best=best,
                 initial=initial, caps=caps, significant=significant, stale=stale, rng=rng.getstate()))
            update(status, status='stopped', layer=layer, updates=last_step)
            return
        atomic_json(folder / 'completed.json', dict(layer=layer, updates=last_step, reason=reason,
                    initial=initial, best=best, joint_training=True, backbone_frozen=True))
        print('completed', layer, best, reason, flush=True)
        del events, models, opts
        gc.collect()
        torch.cuda.empty_cache()
    update(status, status='completed', layers=layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--layers', nargs='*', type=int)
    a = p.parse_args()
    root, m = config(a.directory)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    assert json.loads((root / 'preflight.json').read_text())['passed']
    try:
        run(root, m, a.layers or m['layers'][rank::world], rank)
    except Exception as e:
        status = root / f'training/rank{rank}.json'
        previous = json.loads(status.read_text()) if status.exists() else {}
        previous.update(status='failed', error=repr(e), traceback=traceback.format_exc())
        previous.pop('updated_at', None)
        update(status, **previous)
        raise


if __name__ == '__main__':
    main()
