"""Detached, resumable phase launcher with explicit failure boundaries."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
PY = ROOT/'streaming_sparse/.venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--through', choices=['P1', 'P3'], default='P1')
    args = p.parse_args()
    root = Path(args.directory).resolve()
    lock = (root/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    child = None
    state = {}

    def status(**values):
        state.update(values, controller_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat())
        temp = root/'status.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(root/'status.json')
        print(json.dumps(state), flush=True)

    def stop(signum, frame):
        (root/'STOP').touch()
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    phases = [
        ('P0_math', 'audit', '0', []),
        ('K_initialization', 'fit', '0,1,2,3,4,5,6,7', []),
        ('K_retraining', 'foundation', '0,1,2,3,4,5,6,7', []),
        ('P0_real', 'baseline', '0,1,2,3,4,5,6,7', ['--pilot']),
        ('P1_baseline', 'baseline', '0,1,2,3,4,5,6,7', []),
    ]
    if args.through == 'P3':
        phases += [('P2_projection', 'labels', '0,1,2,3,4,5,6,7', ['--projection']),
                   ('P2_features', 'labels', '0,1,2,3,4,5,6,7', []),
                   ('P3_training', 'train', '0,1,2,3,4,5,6,7', []),
                   ('P3_report', 'report', '0,1,2,3,4,5,6,7', []),
                   ('P3_merge', 'report', '0', ['--merge'])]
    protocol=read(root/'experiment.json')
    if protocol.get('training_objective')=='actual_av_nmse':
        assert args.through=='P3'
        inherited=read(root/'foundation_inheritance.json')
        assert inherited and not inherited['action_heads_inherited']
        for name,digest in inherited['parameters'].items():
            assert hashlib.sha256((root/'parameters'/name).read_bytes()).hexdigest()==digest
        phases=[('P0_math','audit','0',[]),
                ('B15_calibration','budget','0,1,2,3,4,5,6,7',[]),
                ('B15_lock','budget','0',['--lock']),
                ('P1_baseline','baseline','0,1,2,3,4,5,6,7',[]),
                ('P2_projection','labels','0,1,2,3,4,5,6,7',['--projection']),
                ('P2_features','labels','0,1,2,3,4,5,6,7',[]),
                ('P3_preflight','check_attention','0',[]),
                ('P3_training','train','0,1,2,3,4,5,6,7',[]),
                ('P3_report','report','0,1,2,3,4,5,6,7',[]),
                ('P3_merge','report','0',['--merge'])]
        if protocol.get('end_to_end',{}).get('enabled'):
            phases=[('N0_runtime_audit','network_audit','0',[]),
                    phases[0],('P0_full_capture','collect','0,1,2,3,4,5,6,7',[]),
                    ('K_initialization','fit','0,1,2,3,4,5,6,7',[]),
                    ('K_retraining','foundation','0,1,2,3,4,5,6,7',[]),*phases[1:],
                    ('N1_seal','video_worker','0',['--seal']),
                    ('N2_videos','video_worker','0,1,2,3,4,5,6,7',[]),
                    ('N3_report','video_report','0',[])]
    try:
        if not protocol.get('end_to_end',{}).get('enabled'):
            assert read(root/'capture_workers/launcher_status.json')['status'] == 'completed'
        for phase, module, gpus, extra in phases:
            if (root/'STOP').exists():
                raise KeyboardInterrupt()
            logdir = root/f'{phase}_workers'
            if read(logdir/'launcher_status.json').get('status') == 'completed':
                continue
            status(status='running', stage=phase, through=args.through)
            snapshot = root/'code_snapshots'/phase
            snapshot.mkdir(parents=True, exist_ok=True)
            identity = {}
            for path in (ROOT/'migration/unified_router').glob('*.py'):
                data = path.read_bytes()
                (snapshot/path.name).write_bytes(data)
                identity[path.name] = hashlib.sha256(data).hexdigest()
            (snapshot/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
            logdir.mkdir(exist_ok=True)
            with (logdir/'controller.log').open('a') as stream:
                child = subprocess.Popen([str(PY), '-u', str(ROOT/'migration/runtime/launch.py'),
                    '--gpus', gpus, '--module', 'migration.unified_router.'+module,
                    '--logdir', str(logdir), '--', '--directory', str(root), *extra],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if (root/'STOP').exists():
                        raise KeyboardInterrupt()
                    time.sleep(3)
                assert child.returncode == 0, f'{phase} failed; see {logdir}'
                assert read(logdir/'launcher_status.json')['status'] == 'completed'
        if protocol.get('end_to_end',{}).get('enabled'):
            status(status='completed',stage='all_finished',through='E2E',report=str(root/'video_review/REPORT.md'))
        else:
            status(status='phase_complete', stage=args.through, through=args.through,
                   next_stage='P2/P3' if args.through == 'P1' else 'P4_network_selection')
    except KeyboardInterrupt:
        status(status='paused')
    except BaseException as error:
        status(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()


if __name__ == '__main__':
    main()
