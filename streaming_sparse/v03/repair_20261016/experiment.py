"""Train-only A/B/C/D causal fits, with fixed data, initialization and minibatches."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
import torch
from .model import RepairRouter
from ..model import RouterConfig
from ..data import SnapshotData
from ..evaluate import source_identity, statistics
from ..metrics import error_parts, partition_diagnostics
from ...attention.block_statistics import exact_attention
from ...capture.dataset_io import atomic_save, query_slices
from ...evaluation.baselines import mass_selection
from ...v02.evaluation_journal import atomic_json

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'streaming_sparse/v03/rounds/20261016_165725'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def intervention_identity():
    h = hashlib.sha256(source_identity().encode())
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def subset(data):
    prompt = json.loads((data.root/'splits.json').read_text())['train'][0]
    selected = []
    for layer in (4, 14, 24):
        for head in (0, 3, 6, 9):
            matches = [v for v, rows in data.groups.items() if
                       (rows[0]['prompt_id'], rows[0]['seed'], rows[0]['layer_id'],
                        rows[0]['head_id'], rows[0]['history_tokens']) == (prompt, 0, layer, head, 9360)]
            if len(matches) != 1:
                raise ValueError((layer, head, matches))
            selected.append(matches[0])
    return selected


def init_model(fusion, arm, device):
    original = torch.load(OLD/f'runs/pilot/{fusion}/seed0/initial.pt', weights_only=True, map_location='cpu')
    torch.manual_seed(0)
    m = RepairRouter(RouterConfig(**original['config']))
    m.load_state_dict(original['router'], strict=True)
    return m.configure(arm).to(device)


def load_events(data, versions, device):
    cache = {}
    for version in versions:
        events = data.load(version, device)
        trimmed = []
        for e in events:
            indices = query_slices(e, 16)[0]
            trimmed.append({**e, 'q':e['q'][indices].clone(), 'o_dense':e['o_dense'][indices].clone(),
                            'query_indices':e['query_indices'][indices].clone()})
        assert all(torch.equal(trimmed[0]['query_indices'], e['query_indices']) for e in trimmed)
        cache[version] = trimmed
    return cache


@torch.no_grad()
def freeze_partition(model, cache):
    frozen, saved = {}, {}
    for version, events in cache.items():
        e = events[0]
        s = model.precompute(e['k_hist'], e['v_hist'], head_id=e['head_id'], layer_id=e['layer_id'])
        s = {k:v.detach() if isinstance(v, torch.Tensor) else v for k,v in s.items()}
        fixed_stats = [statistics(e['q'], e, s['hard']) for e in events]
        frozen[version] = dict(state=s, stats=fixed_stats)
        saved[version] = dict(labels=s['hard'].argmax(-1).cpu(), raw_labels=s['raw_labels'].cpu(),
                              layer_id=e['layer_id'], head_id=e['head_id'])
    return frozen, saved


def state_for(model, version, events, frozen):
    e = events[0]
    return (model.fixed_precompute(e, frozen[version]['state']) if model.arm in ('A', 'B') else
            model.precompute(e['k_hist'], e['v_hist'], head_id=e['head_id'], layer_id=e['layer_id']))


def stat_for(model, version, i, e, state, frozen):
    return (frozen[version]['stats'][i] if model.arm in ('A', 'B') else
            statistics(e['q'], e, state['assignment']))


def error(output, target):
    se, en = error_parts(output, target)
    return dict(sse=float(se), energy=float(en), error=float(se/(en+1e-8)))


def macro(rows, method):
    groups = defaultdict(list)
    for row in rows:
        groups[f"L{row['layer_id']}H{row['head_id']}"].append(row[method])
    pairs = {k:sum(v['sse'] for v in values)/(sum(v['energy'] for v in values)+1e-8)
             for k,values in groups.items()}
    return dict(macro_pair_relative_error=sum(pairs.values())/len(pairs), per_pair=pairs,
                mean_case_error=sum(r[method]['error'] for r in rows)/len(rows))


@torch.no_grad()
def evaluate(model, cache, frozen, previous=None):
    from scipy.optimize import linear_sum_assignment
    rows, partitions, diagnostics = [], {}, []
    for version, events in cache.items():
        s = state_for(model, version, events, frozen)
        labels = s['hard'].argmax(-1)
        initial = frozen[version]['state']['hard'].argmax(-1)
        b = s['hard'].shape[1]
        counts = torch.bincount(initial*b+labels, minlength=b*b).reshape(b,b).cpu().numpy()
        a, c = linear_sum_assignment(counts, maximize=True)
        change = 1-float(counts[a,c].sum())/len(labels)
        diag = partition_diagnostics(s)
        diag['partition_change_from_initial_best_label_match'] = change
        diag['raw_label_change_from_initial'] = float((labels != initial).float().mean())
        if previous is not None:
            old = previous[version].to(labels.device)
            counts = torch.bincount(old*b+labels, minlength=b*b).reshape(b,b).cpu().numpy()
            a,c = linear_sum_assignment(counts, maximize=True)
            diag['partition_change_since_evaluation_best_label_match'] = 1-float(counts[a,c].sum())/len(labels)
        if model.arm in 'AB':
            assert torch.equal(labels, initial), 'Fixed topology changed'
        partitions[version] = labels.cpu()
        diagnostics.append(dict(cache_version=version, layer_id=events[0]['layer_id'],
                                head_id=events[0]['head_id'], **diag))
        for i,e in enumerate(events):
            stat = stat_for(model, version, i, e, s, frozen)
            route = model.route(e['q'], s, .2)
            mass = mass_selection(stat, s['hard'].sum(0), route['budget'])
            base = frozen[version]['stats'][i]
            initial_mass = mass_selection(base, frozen[version]['state']['hard'].sum(0), route['budget'])
            costs = s['hard'].sum(0)
            kept = int((costs*route['hard_mask']).sum())
            assert kept <= route['budget']
            row = dict(cache_version=version, prompt_id=e['prompt_id'], layer_id=e['layer_id'],
                       head_id=e['head_id'], step_id=e['step_id'], query_indices=e['query_indices'].cpu().tolist(),
                       learned=error(stat.output(route['hard_mask']), e['o_dense']),
                       soft=error(stat.output(route['soft_mask']), e['o_dense']),
                       mass=error(stat.output(mass), e['o_dense']),
                       initial_mass=error(base.output(initial_mass), e['o_dense']),
                       kept_tokens=kept, token_budget=route['budget'],
                       mass_kept_tokens=int((costs*mass).sum()),
                       budget_utilization=kept/max(route['budget'],1))
            rows.append(row)
    results = {m:macro(rows,m) for m in ('learned','soft','mass','initial_mass')}
    ratio = results['learned']['macro_pair_relative_error']/results['mass']['macro_pair_relative_error']
    results.update(learned_over_same_partition_mass=ratio,
                   learned_over_initial_mass=results['learned']['macro_pair_relative_error']/results['initial_mass']['macro_pair_relative_error'],
                   fitted_to_mass_10pct=ratio <= 1.10,
                   pair_count_within_25pct_mass=sum(results['learned']['per_pair'][p] <= 1.25*value
                                                   for p,value in results['mass']['per_pair'].items()),
                   mean_repair_fraction=sum(d['repair_fraction'] for d in diagnostics)/len(diagnostics),
                   mean_partition_change_from_initial=sum(d['partition_change_from_initial_best_label_match'] for d in diagnostics)/len(diagnostics),
                   budget_violations=0, rows=rows, partition_diagnostics=diagnostics)
    return results, partitions


def gradient_groups(model):
    return {name:math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None))
            for name,module in (('feature',model.features), ('aggregator',model.aggregator),
                                ('encoder',model.encoder), ('decoder',model.decoder), ('score',model.decoder.score))}


def fit(fusion, arm, cache, frozen, schedule, out, identity, args):
    directory = out/'runs'/fusion/arm
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'final.pt').exists() and args.resume:
        value = torch.load(directory/'final.pt', map_location='cpu', weights_only=True)
        assert value['identity'] == identity
        return
    if (directory/'initial.pt').exists() and not args.resume:
        raise FileExistsError(directory)
    m = init_model(fusion, arm, args.device)
    trainable = [p for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=.01)
    initial_feature = {k:v.detach().clone() for k,v in m.features.state_dict().items()}
    begin = 0

    def save(path, updates):
        atomic_save(dict(config=asdict(m.config), arm=arm, router=m.state_dict(), optimizer=optimizer.state_dict(),
                         updates=updates, identity=identity, torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                         checkpoint_selection='fixed final update; train-only diagnosis, no Val selection'), path)

    if args.resume and (directory/'latest.pt').exists():
        value = torch.load(directory/'latest.pt', map_location=args.device, weights_only=True)
        assert value['identity'] == identity
        m.load_state_dict(value['router'])
        optimizer.load_state_dict(value['optimizer'])
        begin = value['updates']
        torch.set_rng_state(value['torch_rng'].cpu())
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda_rng']])
        for name in ('metrics.jsonl', 'curve.jsonl'):
            p = directory/name
            if p.exists():
                rows = [json.loads(line) for line in p.read_text().splitlines()]
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['updates'] <= begin))
    else:
        save(directory/'initial.pt', 0)
        report, previous = evaluate(m, cache, frozen)
        atomic_json(directory/'evaluation_000000.json', report)
    previous = None
    atomic_json(directory/'config.json', dict(**identity, fusion=fusion, arm=arm,
                                             parameters=sum(p.numel() for p in m.parameters()),
                                             trainable_parameters=sum(p.numel() for p in trainable)))
    started = time.perf_counter()
    for step in range(begin, args.steps):
        m.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        versions = schedule[step]
        for version in versions:
            events = cache[version]
            state = state_for(m, version, events, frozen)
            losses = []
            for i,e in enumerate(events):
                route = m.route(e['q'], state, .2)
                kept = int((state['hard'].sum(0)*route['hard_mask']).sum())
                if kept > route['budget']:
                    raise AssertionError('Budget exceeded')
                stat = stat_for(m, version, i, e, state, frozen)
                se,en = error_parts(stat.output(route['mask']), e['o_dense'])
                losses.append(se/(en+1e-8))
            loss = torch.stack(losses).mean()/len(versions)
            if not torch.isfinite(loss):
                raise FloatingPointError('Attention loss is not finite')
            loss.backward()
            loss_value += float(loss.detach())
        grad = gradient_groups(m) if step == 0 or (step+1)%25 == 0 else None
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
        optimizer.step()
        if arm == 'A':
            assert all(torch.equal(v, initial_feature[k]) for k,v in m.features.state_dict().items())
        row = dict(updates=step+1, attention_loss=loss_value, gradient_norm=norm,
                   gradient_groups=grad, keep=.2, snapshots=versions,
                   elapsed_seconds=time.perf_counter()-started)
        with (directory/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if step == 0 or (step+1)%25 == 0 or step+1 == args.steps:
            print(json.dumps(dict(fusion=fusion, arm=arm, **{k:v for k,v in row.items() if k != 'snapshots'})), flush=True)
        if (step+1)%100 == 0 or step+1 == args.steps:
            report, previous = evaluate(m, cache, frozen, previous)
            atomic_json(directory/f'evaluation_{step+1:06d}.json', report)
            summary = {k:v for k,v in report.items() if k not in ('rows','partition_diagnostics')}
            summary['updates'] = step+1
            with (directory/'curve.jsonl').open('a') as stream:
                stream.write(json.dumps(summary)+'\n')
            save(directory/'latest.pt', step+1)
            atomic_json(out/'status.json', dict(status='running', fusion=fusion, arm=arm, updates=step+1,
                                               total_updates=args.steps, updated_utc=utc(), gpu='0', max_gpu_jobs=1))
            print(json.dumps(dict(fusion=fusion, arm=arm, updates=step+1,
                                  full_subset_learned=summary['learned']['macro_pair_relative_error'],
                                  same_partition_mass=summary['mass']['macro_pair_relative_error'],
                                  learned_over_mass=summary['learned_over_same_partition_mass'])), flush=True)
    assert arm != 'A' or all(torch.equal(v, initial_feature[k]) for k,v in m.features.state_dict().items())
    save(directory/'final.pt', args.steps)
    del m,optimizer
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()


@torch.no_grad()
def preflight(cache, frozen, fusion, out, device):
    reference = init_model(fusion,'C',device)
    errors = []
    for arm in 'ABCD':
        model = init_model(fusion,arm,device)
        for version,events in cache.items():
            state = state_for(model,version,events,frozen)
            original = frozen[version]['state']
            torch.testing.assert_close(state['hard'],original['hard'],rtol=0,atol=0)
            for e in events:
                route = model.route(e['q'],state,.2)
                expected = reference.route(e['q'],original,.2)
                torch.testing.assert_close(route['scores'],expected['scores'],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(route['hard_mask'],expected['hard_mask'],atol=0,rtol=0)
                stat = statistics(e['q'],e,state['hard'])
                torch.testing.assert_close(stat.dense(),e['o_dense'].float(),atol=5e-5,rtol=5e-5)
                selected = state['hard']@route['hard_mask'] > 0
                direct = exact_attention(e['q'],torch.cat((e['k_hist'][selected],e['k_current'])),
                                         torch.cat((e['v_hist'][selected],e['v_current'])))
                torch.testing.assert_close(stat.output(route['hard_mask']),direct,atol=5e-5,rtol=5e-5)
                errors.append(float((stat.dense()-e['o_dense'].float()).abs().max()))
        del model
    atomic_json(out/f'preflight_{fusion}.json',dict(passed=True,initial_hard_partitions_identical=True,
        effective_score_initialization_matched=True, initial_hard_masks_identical=True,
        dense_and_raw_gather_passed=True,cases_checked=len(errors), max_dense_absolute_difference=max(errors)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(OLD/'data/stage_a'))
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--fusions',nargs='+',default=['concat','film'])
    p.add_argument('--arms',nargs='+',default=['A','B','C','D'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    args = p.parse_args()
    out = Path(args.output).resolve()
    if args.steps < 1 or not set(args.arms) <= set('ABCD') or not set(args.fusions) <= {'concat','film'}:
        raise ValueError('Invalid experiment specification')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.device.startswith('cuda'):
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 1, 'Expose only GPU 0 for this round'
    data = SnapshotData(args.data,'train')
    versions = subset(data)
    selected_records = [r for v in versions for r in data.groups[v]]
    tensor_paths = sorted({data.root/r[k] for r in selected_records for k in ('history','event')})
    files = [dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in tensor_paths]
    rng = random.Random(20261016)
    schedule = []
    while len(schedule) < args.steps:
        order = list(versions)
        rng.shuffle(order)
        schedule.extend([order[i:i+4] for i in range(0,12,4)])
    schedule = schedule[:args.steps]
    identity = dict(protocol='v03_causal_repair_20261016',original_source_sha256=source_identity(),
        intervention_source_sha256=intervention_identity(),steps=args.steps,seed=0,keep_ratios=[.2],
        fixed_query_tile_index=0, events_per_snapshot=4, snapshots_per_update=4,
        schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        subset_file_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        lr=1e-4,weight_decay=.01,grad_clip=1.,loss='attention_output_only',selection='final_update_train_only',
        frozen_backbone=True,val_used=False,test_used=False, physical_heads=[0,3,6,9],layers=[4,14,24],
        torch=str(torch.__version__),cuda_runtime=torch.version.cuda,python=sys.version,
        gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu',
        fit_diagnostic_threshold='final macro pair error <= 1.10 x same-partition Mass; not a generalization gate',
        arms=dict(A='fixed initial F and repaired partition; train aggregator/H/Q decoder/score',
                  B='fixed initial repaired partition; train F and downstream',
                  C='original joint trainable F/partition and downstream',
                  D='C with independent full effective pair score head; all other modules unchanged'))
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == identity, 'Protocol changed; create a new round'
    atomic_json(out/'protocol.json',identity)
    atomic_json(out/'subset.json',dict(split='train',records=selected_records,tensor_files=files))
    atomic_json(out/'schedule.json',schedule)
    cache = load_events(data,versions,args.device)
    try:
        for fusion in args.fusions:
            initial = init_model(fusion,'C',args.device)
            frozen,saved = freeze_partition(initial,cache)
            atomic_save(saved,out/f'initial_partitions_{fusion}.pt')
            preflight(cache,frozen,fusion,out,args.device)
            if not args.preflight_only:
                for arm in args.arms:
                    fit(fusion,arm,cache,frozen,schedule,out,identity,args)
            del initial,frozen
        atomic_json(out/'status.json',dict(status='preflight_passed' if args.preflight_only else 'completed',
            completed_utc=utc(),steps_per_arm=args.steps,fusions=args.fusions,arms=args.arms,
            max_gpu_jobs=1, gpu='0', full_12_head_started=False,diffusion_training_started=False))
    except BaseException:
        atomic_json(out/'status.json',dict(status='failed',updated_utc=utc(),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
