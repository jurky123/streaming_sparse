"""Two exclusive GPUs, validation-only conditional stages, fail-closed execution."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .common import (HERE, DEFAULT_DATA, SnapshotData, protocol, source_identity, atomic_json,
                     protect_output, fixed_diagnostic_samples)
from .model import Router
from .progress import render


def prepare():
    python = str(HERE.parent / '.venv/bin/python'); cwd = str(HERE.parents[1]); jobs = []
    def add(name, module, args, deps=(), gpu='auto', condition=None, run_directory=None):
        jobs.append(dict(id=name, argv=[python, '-u', '-m', module, *map(str, args)], depends_on=list(deps),
                         gpu=gpu, condition=condition, run_directory=str(run_directory) if run_directory else None))
    add('unit_correctness', 'unittest', ['streaming_sparse.v06.test_correctness', 'streaming_sparse.v05.test_correctness', '-v'], gpu=None)
    for gpu in (0, 1):
        add(f'correctness_gpu{gpu}', 'streaming_sparse.v06.sanity',
            ['--output', HERE / f'reports/correctness_gpu{gpu}.json'], ['unit_correctness'], gpu=gpu)
    gate_deps = ['correctness_gpu0', 'correctness_gpu1']
    for seed in (0, 1, 2):
        run = HERE / f'runs/seed{seed}/stage1'
        add(f'C0_seed{seed}', 'streaming_sparse.v06.train', ['--stage', 'stage1', '--seed', seed, '--gate', '{GATE}'], gate_deps, run_directory=run)
    for seed in (0, 1, 2):
        run = HERE / f'runs/seed{seed}/stage2'
        add(f'corrections_seed{seed}', 'streaming_sparse.v06.train', ['--stage', 'stage2', '--seed', seed, '--gate', '{GATE}'],
            [f'C0_seed{seed}'], run_directory=run)
        add(f'core_full_val_seed{seed}', 'streaming_sparse.v06.evaluate',
            ['--stage2', run, '--split', 'val', '--output', run / 'full_val'], [f'corrections_seed{seed}'], run_directory=run)
    add('validation_decision', 'streaming_sparse.v06.report', ['--decide-stage3'],
        [f'core_full_val_seed{s}' for s in (0, 1, 2)], gpu=None)
    terminal = []
    for seed in (0, 1, 2):
        base = HERE / f'runs/seed{seed}'; run = base / 'stage2'
        name = f'core_development_test_seed{seed}'
        add(name, 'streaming_sparse.v06.evaluate', ['--stage2', run, '--split', 'test', '--output', run / 'full_test'],
            ['validation_decision'], run_directory=run); terminal.append(name)
        name = f'offline_fixed_queries_seed{seed}'
        add(name, 'streaming_sparse.v06.diagnostics', ['--stage2', run, '--output', base / 'offline_diagnostics'],
            ['validation_decision']); terminal.append(name)
        for kind in ('C1', 'C2', 'C3'):
            condition = dict(eligible=kind)
            previous = 'validation_decision'
            for stage in ('stage3', 'scratch'):
                directory = base / f'{stage}_{kind}'; name = f'{stage}_{kind}_seed{seed}'
                add(name, 'streaming_sparse.v06.train', ['--stage', stage, '--kind', kind, '--seed', seed, '--gate', '{GATE}'],
                    [previous], condition=condition, run_directory=directory)
                train_job = name
                for split in ('val', 'test'):
                    eval_name = f'{name}_eval_{split}'
                    add(eval_name, 'streaming_sparse.v06.evaluate', ['--checkpoint', directory / f'best_{kind}.pt',
                        '--split', split, '--output', directory / f'full_{split}'], [name], condition=condition, run_directory=directory)
                    terminal.append(eval_name)
                previous = train_job
        run = base / 'signed'; name = f'signed_retraining_seed{seed}'; condition = dict(signed_retraining=True)
        add(name, 'streaming_sparse.v06.train', ['--stage', 'signed', '--seed', seed, '--gate', '{GATE}'],
            ['validation_decision'], condition=condition, run_directory=run)
        for split in ('val', 'test'):
            eval_name = f'{name}_eval_{split}'
            add(eval_name, 'streaming_sparse.v06.evaluate', ['--stage2', run, '--variants', '--split', split,
                '--output', run / f'full_{split}'], [name], condition=condition, run_directory=run); terminal.append(eval_name)
    add('final_review', 'streaming_sparse.v06.report', [], terminal, gpu=None)
    data = SnapshotData(DEFAULT_DATA, 'train'); val = SnapshotData(DEFAULT_DATA, 'val')
    plan = dict(source_sha256=source_identity(), cwd=cwd, jobs=jobs, protocol=protocol(),
                train_snapshots=len(data.groups), validation_snapshots=len(val.groups),
                data_fingerprint=data.fingerprint(), fixed_offline_queries=fixed_diagnostic_samples(val),
                capacity=Router(arms=('C1', 'C2', 'C3')).capacity(),
                environment=dict(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                                 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
    target = HERE / 'reports/plan.json'
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError('Existing different preregistered plan; do not silently replace')
    atomic_json(target, plan)
    return plan


def permitted(condition):
    if condition is None:
        return True
    path = HERE / 'reports/stage2_decision.json'
    if not path.exists():
        raise ValueError('Conditional task has no locked Validation decision')
    decision = json.loads(path.read_text())
    if decision['source_sha256'] != source_identity() or not decision['no_test_reads']:
        raise ValueError('Invalid progression decision')
    if 'eligible' in condition:
        return condition['eligible'] in decision['eligible']
    return decision['signed_retraining_required']


def execute(plan):
    if plan['source_sha256'] != source_identity():
        raise ValueError('Source changed after preregistration')
    output = protect_output(HERE / 'reports/workflow')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing workflow; refuse duplicate workers')
    output.mkdir(parents=True, exist_ok=True)
    snapshot = HERE / 'reports/launched_source'; snapshot.mkdir(exist_ok=True)
    for path in HERE.glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(HERE / 'configs/protocol.json', snapshot / 'protocol.json')
    jobs = plan['jobs']; state = {j['id']: dict(status='pending', gpu=j['gpu'], run_directory=j['run_directory']) for j in jobs}
    running = {}; finished = set(); aborted = False
    def save():
        atomic_json(output / 'status.json', dict(source_sha256=plan['source_sha256'], jobs=state, aborted=aborted,
            updated_utc=datetime.now(timezone.utc).isoformat(), workflow_pid=os.getpid()))
        (HERE / 'reports/STATUS.txt').write_text(render())
    save(); last_save = time.monotonic()
    try:
        while len(finished) < len(jobs):
            for name, (proc, log) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close(); del running[name]
                state[name].update(status='completed' if code == 0 else 'failed', exit_code=code,
                                   finished_utc=datetime.now(timezone.utc).isoformat())
                if code == 0:
                    finished.add(name)
                else:
                    aborted = True
                save()
            if not aborted:
                occupied = {state[n]['gpu'] for n in running}
                for job in jobs:
                    name = job['id']
                    if state[name]['status'] != 'pending' or not set(job['depends_on']) <= finished:
                        continue
                    if not permitted(job['condition']):
                        state[name].update(status='skipped', reason='Predeclared Validation eligibility condition not met')
                        finished.add(name); save(); continue
                    gpu = job['gpu']
                    if gpu == 'auto':
                        gpu = next((i for i in (0, 1) if i not in occupied), 'busy')
                    if gpu == 'busy' or gpu is not None and gpu in occupied or len(running) >= 3:
                        continue
                    env = dict(os.environ)
                    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
                        env.pop(key, None)
                    env.update(plan['environment']); env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
                    gate = str(HERE / f'reports/correctness_gpu{gpu}.json')
                    argv = [item.replace('{GATE}', gate) for item in job['argv']]
                    log = (output / f'{name}.log').open('a')
                    proc = subprocess.Popen(argv, cwd=plan['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT)
                    running[name] = (proc, log); state[name].update(status='running', gpu=gpu, pid=proc.pid,
                        started_utc=datetime.now(timezone.utc).isoformat()); occupied.add(gpu); save()
            if aborted and not running:
                break
            if not running and len(finished) == len(jobs):
                break
            if not running and not aborted:
                raise RuntimeError('DAG deadlock: pending tasks without runnable dependency')
            if time.monotonic() - last_save > 15:
                save(); last_save = time.monotonic()
            time.sleep(.5)
    finally:
        for name, (proc, log) in running.items():
            if proc.poll() is None:
                proc.terminate()
                state[name].update(status='failed', reason='Workflow interrupted')
            log.close()
        save()
    if aborted:
        raise RuntimeError('A task failed; no further tasks launched. Inspect reports/STATUS.txt and logs.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); plan = prepare() if args.prepare else json.loads((HERE / 'reports/plan.json').read_text())
    if args.execute:
        execute(plan)
    else:
        print(json.dumps(dict(source_sha256=plan['source_sha256'], jobs=len(plan['jobs']),
                             capacity=plan['capacity'], train_snapshots=plan['train_snapshots'],
                             validation_snapshots=plan['validation_snapshots']), indent=2))


if __name__ == '__main__':
    main()
