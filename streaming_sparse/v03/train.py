"""Attention-output-only training; no capture, oracle supervision, or auto-resume."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from importlib.metadata import version as package_version
from pathlib import Path
import random
import sys
import time
import torch
from .model import RouterConfig, SparseRouter
from .data import SnapshotData
from .evaluate import evaluate, source_identity, statistics
from .metrics import decoder_diagnostics, error_parts, gradient_gate, mask_change, partition_diagnostics
from .paths import protect_output
from ..capture.dataset_io import atomic_save, query_slices
from ..v02.evaluation_journal import atomic_json


def check_config(config):
    if config.get('loss', 'attention_output_only') != 'attention_output_only':
        raise ValueError('v0.3 trains on attention output error only')
    if any(config.get(k, 0) for k in ('collapse_weight', 'oracle_weight', 'auxiliary_weight')):
        raise ValueError('Additional training objectives are forbidden in this protocol')
    if config.get('snapshots_per_update', 4) != 4 or config.get('tiles_per_event', 1) != 1:
        raise ValueError('Main training requires 4 snapshots and one tile per event')
    if config.get('keep_ratios', [.3, .2]) != [.3, .2] or config.get('selection_keep', .2) != .2:
        raise ValueError('Train budgets must be [.3,.2]; checkpoint selection uses .2')
    if config.get('steps', 1500) < 1 or min(config.get('save_every', 250), config.get('validate_every', 500), config.get('log_every', 50)) < 1:
        raise ValueError('Training/validation/save intervals must be positive')


def provenance(config, data, val, model):
    protocol = Path(__file__).parent/'configs/protocol.json'
    package = Path(__file__).resolve().parents[2]/'restoration_20261016/package_identity.json'
    return dict(version='0.3', source_sha256=source_identity(), protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                package_identity=json.loads(package.read_text()) if package.exists() else None,
                data_fingerprint=data.fingerprint(), validation_data_fingerprint=val.fingerprint() if val else None,
                python=sys.version, torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                packages={name: package_version(name) for name in ('numpy', 'scipy', 'matplotlib', 'torchvision')},
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                parameter_counts=model.parameter_counts(), complete_data_protocol=data.require_complete,
                config=config, checkpoint_selection_split='val', checkpoint_selection_keep=.2,
                checkpoint_selection_metric='macro_prompt_head_relative_error')


def run(config, data_root, output, device='cuda', resume=None, overfit_version=None):
    check_config(config)
    if config.get('stage') == 'overfit' and not overfit_version:
        raise ValueError('Overfit mode requires an explicit --overfit-snapshot from train')
    c = RouterConfig(**config.get('router', {}))
    heads, layers = config.get('heads', [0, 3, 6, 9]), config.get('layers', [4, 14, 24])
    if not set(heads) <= set(c.heads) or not set(layers) <= set(c.layers):
        raise ValueError('Data layer/head IDs must be supported by the router')
    # Fail before creating run directories or touching GPU when real tensors are missing.
    data = SnapshotData(data_root, 'train', heads, layers, config.get('require_complete', True))
    # Single-snapshot overfit is a train-only gate, before full data collection.
    val = None if overfit_version else SnapshotData(data_root, 'val', heads, layers, config.get('require_complete', True))
    if overfit_version is not None and overfit_version not in data.groups:
        raise ValueError('Overfit snapshot must belong to train, never val/test')
    if config.get('require_complete', True) and not overfit_version:
        from .prerequisites import require_training_gates
        require_training_gates(config, c.fusion)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not start training')
    output = protect_output(output)
    if not resume and output.exists() and any(p.name != '.gitkeep' for p in output.iterdir()):
        raise FileExistsError('Use a new run directory or explicitly resume latest.pt')
    rng = random.Random(config.get('seed', 0))
    torch.manual_seed(config.get('seed', 0))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SparseRouter(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=config.get('weight_decay', .01))
    run_identity = provenance(config, data, val, model)
    run_identity['overfit_version'] = overfit_version
    model.run_identity = run_identity
    begin, best, validated_updates = 0, float('inf'), 0
    last_diagnostics = []
    if resume:
        value = torch.load(resume, map_location=device, weights_only=True)
        if value['run_identity'] != run_identity or value['config'] != asdict(c):
            raise ValueError('Resume source/config/data/environment identity mismatch')
        model.load_state_dict(value['router'], strict=True)
        optimizer.load_state_dict(value['optimizer'])
        rng.setstate(value['python_rng'])
        torch.set_rng_state(value['torch_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all([x.cpu() for x in value['cuda_rng']])
        begin, best = value['updates'], value['best_val']
        validated_updates = value['validated_updates']
        last_diagnostics = value['diagnostics']
        if not output.exists():
            raise ValueError('Resume requires the original run directory and validation history')
        for name in ('metrics.jsonl', 'validation_curve.jsonl'):
            path = output/name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['updates'] <= begin))
    else:
        output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'config.json', config)
    atomic_json(output/'provenance.json', run_identity)
    atomic_json(output/'validation_subset.json', val.validation_versions() if val else [])

    def checkpoint(updates, diagnostics):
        return dict(**model.checkpoint(), optimizer=optimizer.state_dict(), updates=updates,
                    best_val=best, python_rng=rng.getstate(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                    run_identity=run_identity, diagnostics=diagnostics, validated_updates=validated_updates,
                    diagnostics_timing='Training forward before the most recent optimizer update')

    def validate(updates):
        nonlocal best, validated_updates
        report = evaluate(model, val, output/'validation'/f'step{updates:06d}',
                          keeps=(.2,), baselines=False, versions=val.validation_versions(), bootstrap=100)
        method = 'B4' if c.fusion == 'concat' else 'B5'
        score = report['metrics'][f'{method}/keep=0.2']['macro_prompt_head_relative_error']
        validated_updates = updates
        if score < best:
            best = score
            atomic_save(dict(**checkpoint(updates, last_diagnostics), selection_score=score,
                             selection_split='val', selection_keep=.2), output/'best.pt')
        path = output/'validation_curve.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        rows = [r for r in rows if r['updates'] != updates]
        rows.append(dict(updates=updates, score=score, keep=.2, split='val'))
        path.write_text(''.join(json.dumps(r)+'\n' for r in sorted(rows, key=lambda r: r['updates'])))
        atomic_save(checkpoint(updates, last_diagnostics), output/'latest.pt')

    if not resume:
        atomic_save(checkpoint(0, []), output/'initial.pt')
        atomic_save(checkpoint(0, []), output/'latest.pt')
    elif val is not None and begin > validated_updates and (begin % config.get('validate_every', 500) == 0 or begin == config.get('steps', 1500)):
        # A stop during validation resumes that journal before any new update.
        validate(begin)
    for step in range(begin, config.get('steps', 1500)):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        versions = [overfit_version] if overfit_version else data.sample_versions(rng, 4)
        keep = .2 if overfit_version else rng.choice([.3, .2])
        diagnose = step == 0 or (step+1) % config.get('log_every', 50) == 0 or (step+1) % config.get('save_every', 250) == 0 or (step+1) % config.get('validate_every', 500) == 0 or step+1 == config.get('steps', 1500)
        diagnostics, loss_total, gradient_report = [], 0., None
        for version in versions:
            events = data.load(version, device)
            first = events[0]
            state = model.precompute(first['k_hist'], first['v_hist'], first['token_positions'], first['head_id'], first['layer_id'])
            state['features'].retain_grad()
            state['probabilities'].retain_grad()
            tile_index = 0 if overfit_version else rng.randrange(len(query_slices(first, 16)))
            expected_indices = None
            losses, masks, decoders = [], [], []
            for event in events:
                selection = query_slices(event, 16)[tile_index]
                query_ids = event['query_indices'][selection]
                if expected_indices is None:
                    expected_indices = query_ids
                elif not torch.equal(expected_indices, query_ids):
                    raise ValueError('Step-wise mask diagnostics require aligned captured query windows')
                q = event['q'][selection]
                route = model.route(q, state, keep)
                stats = statistics(q, event, state['assignment'])
                se, energy = error_parts(stats.output(route['mask']), event['o_dense'][selection])
                losses.append(se/(energy+1e-8))
                masks.append((state['hard'].detach()@route['hard_mask'] > 0))
                if int((state['hard'].sum(0)*route['hard_mask']).sum()) > route['budget']:
                    raise AssertionError('Token budget exceeded')
                if diagnose:
                    with torch.no_grad():
                        decoders.append(dict(step_id=event['step_id'], **decoder_diagnostics(route, stats, event['o_dense'][selection], state['hard'])))
            loss = torch.stack(losses).mean()/len(versions)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite attention loss')
            if step == 0:
                # Isolate the gradient through clustering from the representation path.
                pg, = torch.autograd.grad(loss, state['probabilities'], retain_graph=True)
                fg, = torch.autograd.grad(state['probabilities'], state['features'], grad_outputs=pg, retain_graph=True)
                state['cluster_feature_path_norm'] = float(fg.norm())
            loss.backward()
            loss_total += float(loss.detach())
            if step == 0:
                gradient_report = gradient_gate(model, state)
            if diagnose:
                diagnostics.append(dict(cache_version=version, layer_id=first['layer_id'], head_id=first['head_id'],
                    partition=partition_diagnostics(state), decoder=decoders,
                    step_mask_jaccard=mask_change(masks), precompute_calls=1, routed_diffusion_steps=4))
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.), error_if_nonfinite=True))
        optimizer.step()
        if diagnose:
            last_diagnostics = diagnostics
        row = dict(updates=step+1, attention_loss=loss_total, keep=keep, snapshots=versions,
                   gradient_norm=gradient_norm, first_update_gradient_gate=gradient_report,
                   diagnostics=diagnostics, elapsed_seconds=time.perf_counter()-start)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if diagnose:
            print(json.dumps({k: v for k, v in row.items() if k not in ('diagnostics', 'snapshots')}), flush=True)
        # Save the durable optimizer state before validation; replay does not retrain updates.
        validate_now = val is not None and ((step+1) % config.get('validate_every', 500) == 0 or step+1 == config.get('steps', 1500))
        if (step+1) % config.get('save_every', 250) == 0 or validate_now or step+1 == config.get('steps', 1500):
            atomic_save(checkpoint(step+1, last_diagnostics), output/'latest.pt')
        if validate_now:
            validate(step+1)
    atomic_save(checkpoint(config.get('steps', 1500), last_diagnostics), output/'final.pt')
    if overfit_version:
        model.eval()
        with torch.no_grad():
            events = data.load(overfit_version, device)
            first = events[0]
            state = model.precompute(first['k_hist'], first['v_hist'], first['token_positions'], first['head_id'], first['layer_id'])
            errors = []
            for event in events:
                selection = query_slices(event, 16)[0]
                q = event['q'][selection]
                route = model.route(q, state, .2)
                se, en = error_parts(statistics(q, event, state['hard']).output(route['hard_mask']), event['o_dense'][selection])
                errors.append(float(se/(en+1e-8)))
        initial_loss = json.loads((output/'metrics.jsonl').read_text().splitlines()[0])['attention_loss']
        final_error = sum(errors)/len(errors)
        atomic_json(output/'overfit_gate.json', dict(initial_hard_error=initial_loss, final_hard_error=final_error,
            passed=final_error < initial_loss,
            real_qkv=data.require_complete or config.get('data_kind') == 'real_capture', cache_version=overfit_version,
            fusion=c.fusion, source_sha256=source_identity(), data_fingerprint=data.fingerprint()))
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int)
    p.add_argument('--shared', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--overfit-snapshot', help='Explicit train snapshot ID; reuses its four steps and first tile')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    if args.shared:
        config.setdefault('router', {})['conditioned'] = False
    run(config, args.data, args.output, args.device, args.resume, args.overfit_snapshot)


if __name__ == '__main__':
    main()
