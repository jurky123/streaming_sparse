"""Sequential fresh-round execution on one visible GPU, stopping at failed gates."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from .data import SnapshotData
from .evaluate import source_identity
from .gate import architecture_gate
from ..v02.evaluation_journal import atomic_json

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class GateNotPassed(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


class Round:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.protocol = json.loads((HERE/'configs/protocol.json').read_text())
        self.source = source_identity()
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
            self.env.pop(key, None)
        self.state = dict(round_root=str(self.root), pid=os.getpid(), started_utc=now(),
                          gpu_index=0, max_simultaneous_gpu_jobs=1, restore_previous_experiments=False,
                          source_sha256=self.source, status='running', phase='initializing', completed=[])
        self.save()

    def save(self):
        self.state['updated_utc'] = now()
        atomic_json(self.root/'status.json', self.state)

    def stopped(self):
        return (self.root/'STOP').exists()

    def job(self, name, module, args):
        if self.stopped():
            raise GateNotPassed('STOP requested; no further jobs started')
        if source_identity() != self.source:
            raise GateNotPassed('Source changed during the round; preserve existing evidence and review before continuing')
        command = [sys.executable, '-u', '-m', module, *map(str, args)]
        log_path = self.root/'logs'/f'{name}.log'
        if log_path.exists():
            raise FileExistsError(f'Fresh execution will not overwrite an existing job log: {log_path}')
        self.state.update(phase=name, log=str(log_path), command=command, child_pid=None)
        self.save()
        print(json.dumps(dict(event='start', name=name, utc=now(), command=command)), flush=True)
        start = time.monotonic()
        with log_path.open('w') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=self.env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            self.state['child_pid'] = process.pid
            self.save()
            while process.poll() is None:
                if self.stopped():
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    raise GateNotPassed('STOP requested; active job terminated and artifacts preserved')
                time.sleep(3)
        result = dict(name=name, exit_code=process.returncode, seconds=time.monotonic()-start,
                      log=str(log_path), finished_utc=now())
        self.state['completed'].append(result)
        self.state['child_pid'] = None
        self.save()
        print(json.dumps(dict(event='finished', **result)), flush=True)
        if process.returncode:
            raise RuntimeError(f'{name} exited with {process.returncode}; inspect {log_path}')

    def capture(self, name, output, offset, limit, seeds, heads):
        self.job(name, 'streaming_sparse.v02.capture', ['--mode', 'capture', '--prompts', self.root/'configs/prompts.txt',
            '--offset', offset, '--limit', limit, '--frames', 12, '--seeds', *seeds,
            '--layers', 4, 14, 24, '--heads', *heads, '--query-tile', 16, '--tiles-per-event', 8,
            '--backend', 'sdpa', '--output', output])

    def config(self, stage, fusion):
        template = f'overfit_{fusion}' if stage == 'overfit' else f'fusion_{fusion}' if stage == 'pilot' else f'full_{fusion}'
        value = json.loads((HERE/'configs'/f'{template}.json').read_text())
        value.update(gate_reports=str(self.root/'reports'), overfit_runs=str(self.root/'runs/overfit'))
        if stage == 'overfit':
            value.update(require_complete=False, data_kind='real_capture', heads=[0], layers=[4])
        path = self.root/'configs'/f'{stage}_{fusion}.json'
        atomic_json(path, value)
        return path

    def audit_data(self, root, heads):
        coverage = {}
        for split in ('train', 'val', 'test'):
            data = SnapshotData(root, split, heads=heads)
            coverage[split] = dict(events=len(data.dataset.records), snapshots=len(data.groups),
                                   fingerprint=data.fingerprint())
        atomic_json(self.root/'reports'/f'coverage_{len(heads)}heads.json', coverage)
        return coverage

    def execute(self):
        stage_a = self.root/'data/stage_a'
        initial_run = json.loads((stage_a/'run.json').read_text())
        rows = [json.loads(line) for line in (stage_a/'events.jsonl').read_text().splitlines()]
        if len(rows) != 144 or len(initial_run['runs']) != 1:
            raise ValueError('Initial capture must be this round\'s one Prompt x seed0 x three layers x four heads')
        if max(r['dense_fp32_vs_backend_error'] for r in rows) > 1e-4:
            raise GateNotPassed('Initial capture backend numerical error is too large')
        atomic_json(self.root/'reports/initial_capture.json', initial_run)
        atomic_json(stage_a/'splits.json', self.protocol['splits'])
        initial = self.root/'data/initial_view'
        initial.mkdir(exist_ok=False)
        for row in rows:
            for field in ('history', 'event'):
                row[field] = '../stage_a/'+row[field]
        (initial/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        atomic_json(initial/'splits.json', self.protocol['splits'])
        snapshot = next(r['cache_version'] for r in rows if r['layer_id'] == 4 and r['head_id'] == 0 and r['history_tokens'] == 4680)
        self.job('real_correctness_initial', 'streaming_sparse.v03.sanity',
            ['--data', initial, '--initial-subset', '--snapshot', snapshot, '--device', 'cuda',
             '--output', self.root/'reports/real_correctness.json'])
        for fusion in ('concat', 'film'):
            self.job(f'overfit_{fusion}', 'streaming_sparse.v03.train',
                ['--config', self.config('overfit', fusion), '--data', initial,
                 '--overfit-snapshot', snapshot, '--output', self.root/'runs/overfit'/fusion])
            report = json.loads((self.root/'runs/overfit'/fusion/'overfit_gate.json').read_text())
            if not report['passed']:
                raise GateNotPassed(f'{fusion} failed single-snapshot hard-error overfit; do not scale up')
        # Finish the new four-head Cartesian product without recapturing the initial cells.
        self.capture('capture_first_prompt_seed1', stage_a, 0, 1, [1], [0, 3, 6, 9])
        self.capture('capture_remaining_23_prompts', stage_a, 1, 23, [0, 1], [0, 3, 6, 9])
        self.audit_data(stage_a, [0, 3, 6, 9])
        self.job('real_correctness_complete_stage_a', 'streaming_sparse.v03.sanity',
            ['--data', stage_a, '--snapshot', snapshot, '--device', 'cuda',
             '--output', self.root/'reports/real_correctness.json'])
        summaries, runs = [], []
        for fusion in ('concat', 'film'):
            run = self.root/'runs/pilot'/fusion/'seed0'
            self.job(f'pilot_{fusion}', 'streaming_sparse.v03.train',
                ['--config', self.config('pilot', fusion), '--data', stage_a, '--output', run])
            self.job(f'pilot_val_{fusion}', 'streaming_sparse.v03.evaluate',
                ['--data', stage_a, '--checkpoint', run/'best.pt', '--output', run/'evaluation_val', '--split', 'val'])
            runs.append(run)
            summaries.append(run/'evaluation_val/summary.json')
        overfits = [self.root/'runs/overfit'/fusion/'overfit_gate.json' for fusion in ('concat', 'film')]
        gate = architecture_gate(summaries, runs, overfits)
        atomic_json(self.root/'reports/architecture_gate.json', gate)
        if not gate['capture_allowed']:
            raise GateNotPassed('4-head architecture gate did not pass; inspect architecture_gate.json')
        extra = self.root/'data/extra_heads'
        self.job('capture_extra_8heads', 'streaming_sparse.v03.capture',
                 ['--output', extra, '--gate', self.root/'reports/architecture_gate.json', '--execute'])
        full = self.root/'data/full'
        self.job('merge_12heads', 'streaming_sparse.v03.merge', ['--sources', stage_a, extra, '--output', full])
        self.audit_data(full, list(range(12)))
        final_reports = []
        for fusion in ('concat', 'film'):
            for seed in (0, 1, 2):
                run = self.root/'runs/full'/fusion/f'seed{seed}'
                self.job(f'full_{fusion}_seed{seed}', 'streaming_sparse.v03.train',
                    ['--config', self.config('full', fusion), '--data', full, '--seed', seed, '--output', run])
                for split in ('val', 'test'):
                    target = run/f'evaluation_{split}'
                    self.job(f'full_{fusion}_seed{seed}_{split}', 'streaming_sparse.v03.evaluate',
                        ['--data', full, '--checkpoint', run/'best.pt', '--output', target, '--split', split,
                         '--heads', *range(12)])
                    final_reports.append(target/'summary.json')
        self.job('final_analysis', 'streaming_sparse.v03.analyze',
            ['--reports', *final_reports, '--require-full-seeds', '--output', self.root/'reports/final_analysis.json'])
        self.state.update(status='complete', phase='complete')
        self.save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Round(root)
        try:
            runner.execute()
        except Exception as error:
            runner.state.update(status='gate_not_passed' if isinstance(error, GateNotPassed) else 'failed',
                                error=str(error), traceback=traceback.format_exc(), child_pid=None)
            runner.save()
            traceback.print_exc()
            raise SystemExit(2)


if __name__ == '__main__':
    main()
