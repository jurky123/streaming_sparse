"""StageI C0; StageII independent frozen-base corrections; gated StageIII/scratch."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import torch
from .common import (HERE, DEFAULT_DATA, SnapshotData, BalancedSchedule, atomic_json, atomic_save,
                     query_slices, source_identity, file_sha, protect_output, protocol, monitor_versions,
                     tensor_digest)
from .model import Router, load_router, from_base
from .evaluate import FrozenCache, evaluate
from .metrics import trace_summary
from ..v05.primitives import statistics


def update(model, data, chosen, optimizer, active, cache=None, trace=False):
    model.train(); optimizer.zero_grad(set_to_none=True)
    numerators = {name: [] for name in active}; energies = []; tile_errors = {name: [] for name in active}
    traces = {}; samples = []; relations = []
    for version, tile in chosen:
        if cache:
            events, state = cache.get(version)
        else:
            events = data.load(version, next(model.parameters()).device); first = events[0]
            state = model.precompute(first['k_hist'], first['v_hist'], head_id=first['head_id'], layer_id=first['layer_id'])
        reference = None
        for event in events:
            selection = query_slices(event, 16)[tile]; indices = event['query_indices'][selection]
            if reference is None:
                reference = indices
            elif not torch.equal(indices, reference):
                raise ValueError('Unaligned query indices across four steps')
            q = event['q'][selection]; target = event['o_dense'][selection].float()
            stat = statistics(q, event, state['assignment'])
            capture = trace and not traces
            routes = model.route(q, state, .2, trace=capture, only=active)
            energy = target.square().sum(); energies.append(energy)
            for name in active:
                route = routes[name]
                if not torch.isfinite(route['scores']).all():
                    raise FloatingPointError('Nonfinite scores')
                se = (stat.output(route['mask']) - target).square().sum()
                numerators[name].append(se); tile_errors[name].append(float((se / (energy + 1e-8)).detach()))
                if capture:
                    traces[name] = trace_summary(route, state)
                    traces[name]['scaled_correction_to_relevance_norm'] = float(route['scaled_correction'].detach().norm() / route['relevance'].detach().norm().clamp_min(1e-12))
                    for layer in route['layers']:
                        r = layer.get('relation')
                        if r is not None and r.requires_grad:
                            r.retain_grad(); relations.append((name, r))
        samples.append(dict(cache_version=version, tile_id=tile, partition_hash=state['partition_hash']))
    energy = torch.stack(energies).sum()
    losses = {name: torch.stack(values).sum() / (energy + 1e-8) for name, values in numerators.items()}
    # StageII has disjoint trainable branches and a fully frozen base. Sum preserves
    # exactly the independent loss/gradient for each arm; clip each arm separately.
    objective = torch.stack(list(losses.values())).sum()
    if not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite attention reconstruction loss')
    objective.backward()
    if model.base_frozen:
        if any(p.grad is not None for module in (model.frontend, model.query_fusion, model.relevance) for p in module.parameters()):
            raise AssertionError('Frozen base received a gradient')
        norms = {name: float(torch.nn.utils.clip_grad_norm_(model.branches[name].parameters(), 1., error_if_nonfinite=True)) for name in active}
    else:
        norms = {'all': float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))}
    gradients = {}
    if trace:
        for name, module in [('frontend', model.frontend), ('query_fusion', model.query_fusion), ('relevance', model.relevance), *model.branches.items()]:
            gs = [p.grad for p in module.parameters() if p.grad is not None]
            gradients[name] = sum(float(g.detach().square().sum()) for g in gs) ** .5
        gradients['relation'] = {name: sum(float(r.grad.square().sum()) for n, r in relations if n == name and r.grad is not None) ** .5 for name in active}
    optimizer.step()
    result = dict(attention_loss={k: float(v.detach()) for k, v in losses.items()},
                  tile_mean_relative_error={k: sum(v) / len(v) for k, v in tile_errors.items()},
                  gradient_norm_before_clip=norms, samples=samples, keep=.2)
    if trace:
        result.update(representation_diagnostics=traces, gradients=gradients,
                      correction_alpha={k: float(model.branches[k].alpha.detach()) for k in model.arms})
    return result


def run(stage, seed, gate, resume=None, kind=None, max_updates=None):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Exactly one visible BF16 CUDA GPU is required')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    p = protocol(); current = source_identity()
    check = json.loads(Path(gate).read_text())
    if not check.get('passed') or not check.get('full_head_coverage') or check['source_sha256'] != current:
        raise ValueError('Current-source real36-pair correctness gate required')
    if check['gpu_name'] != torch.cuda.get_device_name(0):
        raise ValueError('Correctness gate GPU mismatch')
    train = SnapshotData(DEFAULT_DATA, 'train'); val = SnapshotData(DEFAULT_DATA, 'val')
    if train.fingerprint() != check['data_fingerprint']:
        raise ValueError('Correctness data fingerprint mismatch')
    random.seed(seed); torch.manual_seed(seed)
    seed_root = HERE / f'runs/seed{seed}'
    baseline_path = seed_root / 'stage1/best_C0.pt'
    source_checkpoint = None; scratch_segments = None
    if stage == 'stage1':
        model = Router(correction_seed=100000 + seed).cuda(); arms = ['C0']
    elif stage in ('stage2', 'signed'):
        names = ('C1', 'C2', 'C3') if stage == 'stage2' else ('C3_nocenter', 'C3_amplitude', 'C3_positive')
        model, _ = from_base(baseline_path, names, seed, 'cuda'); model.freeze_base(); arms = list(names)
        source_checkpoint = file_sha(baseline_path)
    elif stage in ('stage3', 'scratch'):
        if kind not in ('C1', 'C2', 'C3'):
            raise ValueError('Joint runs require a core correction kind')
        decision = json.loads((HERE / 'reports/stage2_decision.json').read_text())
        if decision['source_sha256'] != current or kind not in decision['eligible']:
            raise ValueError('Validation has not authorized this structure for StageIII')
        if stage == 'stage3':
            path = seed_root / f'stage2/best_{kind}.pt'
            selected, saved = load_router(path, 'cuda')
            model, _ = from_base(baseline_path, (kind,), seed, 'cuda')
            model.branches[kind].load_state_dict(selected.branches[kind].state_dict()); source_checkpoint = file_sha(path)
        else:
            # Fresh base, not StageI best, paired with exact StageII initial correction.
            model, _ = from_base(seed_root / 'stage1/initial.pt', (kind,), seed, 'cuda')
            one = json.loads((seed_root / 'stage1/status.json').read_text())['updates']
            two = json.loads((seed_root / 'stage2/status.json').read_text())['arm_updates'][kind]
            three = json.loads((seed_root / f'stage3_{kind}/status.json').read_text())['updates']
            scratch_segments = [one, two, three]
        arms = [kind]
    else:
        raise ValueError(stage)
    settings = dict(p['stage2' if stage == 'signed' else stage])
    if stage == 'scratch':
        settings.update(max_updates=sum(scratch_segments), min_updates=sum(scratch_segments))
    if max_updates is not None:
        # Only correctness/smoke jobs use this; formal workflow never passes an override.
        settings.update(max_updates=max_updates, min_updates=max_updates, validate_every=max_updates)
    directory = stage + ('_' + kind if kind else '')
    if max_updates is not None:
        directory = 'smoke_' + directory
    output = protect_output(seed_root / directory)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    initial_base = model.base_digest()
    correction_initial = {k: tensor_digest(model.branches[k].state_dict().items()) for k in model.arms}
    identity = dict(stage=stage, seed=seed, kind=kind, config=settings, source_sha256=current,
                    data_fingerprint=train.fingerprint(), initial_base_digest=initial_base,
                    initial_correction_digests=correction_initial, source_checkpoint_sha256=source_checkpoint,
                    capacity=model.capacity(), torch=str(torch.__version__), python=sys.version, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    scratch_matched_schedule_segments=scratch_segments)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=settings['lr'], weight_decay=.01)
    schedule = BalancedSchedule(sorted(train.groups), seed)
    begin = 0; best = {k: float('inf') for k in arms}; best_step = {k: 0 for k in arms}
    reference = dict(best); stale = {k: 0 for k in arms}; arm_updates = {k: 0 for k in arms}; active = list(arms); history = []
    if resume:
        saved = torch.load(resume, map_location='cuda', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Resume identity/source/config mismatch')
        model.load_state_dict(saved['router']); optimizer.load_state_dict(saved['optimizer']); schedule.load_state_dict(saved['sampler'])
        begin = saved['updates']; best = saved['best']; best_step = saved['best_step']; reference = saved['plateau_reference']
        stale = saved['stale']; active = saved['active']; arm_updates = saved['arm_updates']; history = saved['validation_history']
        torch.set_rng_state(saved['torch_rng'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']]); random.setstate(saved['python_rng'])
        metric_file = output / 'metrics.jsonl'
        if metric_file.exists():
            lines = [json.loads(line) for line in metric_file.read_text().splitlines()]
            metric_file.write_text(''.join(json.dumps(line) + '\n' for line in lines if line['updates'] <= begin))
    atomic_json(output / 'provenance.json', identity)
    cache = FrozenCache(model, train, 'cuda') if model.base_frozen else None

    def status(label, step, **extra):
        atomic_json(output / 'status.json', dict(status=label, stage=stage, seed=seed, kind=kind, updates=step,
            max_updates=settings['max_updates'], active=active, arm_updates=arm_updates,
            best_validation={k: v if math.isfinite(v) else None for k, v in best.items()}, best_step=best_step,
            updated_utc=datetime.now(timezone.utc).isoformat(), **extra))

    def save(path, step):
        atomic_save(dict(**model.checkpoint(), identity=identity, updates=step, optimizer=optimizer.state_dict(),
            sampler=schedule.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), best=best, best_step=best_step, plateau_reference=reference,
            stale=stale, active=active, arm_updates=arm_updates, validation_history=history), path)

    def validate(step):
        if model.base_frozen and model.base_digest() != initial_base:
            raise AssertionError('Frozen C0 base changed')
        status('validating', step)
        path = output / 'validation_candidates' / f'update{step:06d}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
        save(path, step)
        summary = evaluate(model, val, monitor_versions(val), output / f'validation/update{step:06d}',
                           checkpoint_sha=file_sha(path), monitor=True)
        metrics = summary['metrics']['keep=0.2']
        values = {name: metrics[name]['global_attention_error'] for name in ['C0', *model.arms]}
        improved = []
        for name in active:
            score = values[name]
            if score < reference[name] * (1 - settings['min_relative_improvement']):
                reference[name] = score; stale[name] = 0
            else:
                stale[name] += 1
            if score < best[name]:
                best[name] = score; best_step[name] = step; improved.append(name)
        history.append(dict(updates=step, hard=values, mass=metrics['Mass']['global_attention_error'],
                            soft={k: metrics[k + '_soft']['global_attention_error'] for k in values},
                            improved=improved, frozen_base_digest=model.base_digest() if model.base_frozen else None))
        atomic_json(output / 'validation_curve.json', history)
        for name in improved:
            save(output / f'best_{name}.pt', step)
        print(json.dumps(dict(event='validation', **history[-1])), flush=True)

    if not resume:
        status('initializing', 0); save(output / 'initial.pt', 0); validate(0)
    final = begin
    for step in range(begin, settings['max_updates']):
        local_step = step; config = settings
        if scratch_segments is not None:
            boundary = 0
            for segment, length in enumerate(scratch_segments):
                if step < boundary + length:
                    local_step = step - boundary
                    config = dict(p[('stage1', 'stage2', 'stage3')[segment]], max_updates=length)
                    if local_step == 0:
                        schedule = BalancedSchedule(sorted(train.groups), seed)
                        optimizer.state.clear()
                    break
                boundary += length
        warmup = config['warmup']; progress = max(0., (local_step - warmup) / max(1, config['max_updates'] - warmup))
        scale = (local_step + 1) / warmup if local_step < warmup else .05 + .95 * .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = config['lr'] * scale
        row = dict(updates=step + 1, lr=optimizer.param_groups[0]['lr'],
                   **update(model, train, schedule.sample(settings['snapshots_per_update']), optimizer, active, cache,
                            trace=step == 0 or (step + 1) % 50 == 0))
        for name in active:
            arm_updates[name] = step + 1
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        final = step + 1; status('training', final)
        if step == 0 or final % 50 == 0:
            print(json.dumps({k: v for k, v in row.items() if k not in ('samples', 'representation_diagnostics')}), flush=True)
        if final % settings['validate_every'] == 0 or final == settings['max_updates']:
            validate(final)
            if stage != 'scratch' and final >= settings['min_updates']:
                active = [name for name in active if stale[name] < settings['patience']]
            save(output / 'latest.pt', final); save(output / f'step{final:06d}.pt', final)
            if not active:
                break
    save(output / 'final.pt', final)
    if model.base_frozen and model.base_digest() != initial_base:
        raise AssertionError('Frozen base drift at completion')
    atomic_json(output / 'training_exposure.json', dict(final_visits=schedule.visits,
                final_all8_tiles=all(v >= 8 for v in schedule.visits.values()),
                min_snapshot_visits=min(schedule.visits.values()), base_frozen=model.base_frozen,
                selected_exposure_note='Each selected checkpoint stores its own sampler state; final exposure is not selected exposure.'))
    status('completed', final, stop_reason='Validation plateau' if final < settings['max_updates'] else
           'Predeclared maximum; inspect Validation trend before claiming convergence')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('stage1', 'stage2', 'signed', 'stage3', 'scratch'), required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--gate', required=True); parser.add_argument('--kind'); parser.add_argument('--resume')
    parser.add_argument('--smoke-updates', type=int)
    args = parser.parse_args(); run(args.stage, args.seed, args.gate, args.resume, args.kind, args.smoke_updates)


if __name__ == '__main__':
    main()
