"""Observe the existing workflow without changing its configuration or processes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time


HOME = Path(__file__).resolve().parent
MODULES = {f'streaming_sparse.v02.{name}' for name in
           ('workflow', 'train', 'evaluate', 'capture', 'report', 'p0',
            'review_pack', 'final_analysis')}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path, value):
    if path.is_symlink():
        path = path.resolve()
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def last_row(path):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 2_000_000))
            lines = stream.read().splitlines(keepends=True)
        for line in reversed(lines):
            if line.endswith(b'\n'):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
    except OSError:
        pass
    return None


def processes():
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [part.decode(errors='replace') for part in
                    (entry / 'cmdline').read_bytes().split(b'\0') if part]
            module = next((args[i + 1] for i, arg in enumerate(args[:-1])
                           if arg == '-m' and args[i + 1] in MODULES), None)
            if module is None:
                continue
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            item = dict(pid=int(entry.name), ppid=int(fields[1]), module=module,
                        cpu_ticks=int(fields[11]) + int(fields[12]), state=fields[0])
            for option in ('--output', '--split'):
                if option in args:
                    item[option[2:]] = args[args.index(option) + 1]
            result.append(item)
        except (OSError, ValueError, IndexError):
            continue
    return sorted(result, key=lambda item: item['pid'])


def run(interval=30, stall_seconds=900):
    logs = HOME / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (HOME / 'monitor.pid').write_text(str(os.getpid()) + '\n')
        last_signature = last_notice = None
        last_activity = time.monotonic()
        missing_since = None
        while True:
            now = time.monotonic()
            state = read_json(HOME / 'status.json') or {}
            active = processes()
            try:
                workflow_pid = int((HOME / 'workflow.pid').read_text())
            except (OSError, ValueError):
                workflow_pid = None
            workflow_alive = any(p['pid'] == workflow_pid and
                                 p['module'].endswith('.workflow') and p['state'] != 'Z'
                                 for p in active)
            own = {workflow_pid}
            for _ in range(len(active)):
                own.update(p['pid'] for p in active if p['ppid'] in own)
            active = [p for p in active if p['pid'] in own]
            progress = []
            for process in active:
                if 'output' not in process:
                    continue
                directory = Path(process['output'])
                if process['module'].endswith('.evaluate'):
                    progress.append(dict(kind='evaluation', path=str(directory),
                                         progress=read_json(directory / 'progress.json')))
                elif process['module'].endswith('.train'):
                    row = last_row(directory / 'metrics.jsonl')
                    progress.append(dict(kind='training', path=str(directory),
                                         updates=row['step'] + 1 if row else 0,
                                         loss=row.get('attention_loss') if row else None))
                elif process['module'].endswith('.capture'):
                    manifest = directory / 'events.jsonl'
                    progress.append(dict(kind='capture', path=str(directory),
                                         manifest_bytes=manifest.stat().st_size if manifest.exists() else 0))
            signature = json.dumps([state, active, progress], sort_keys=True)
            if signature != last_signature:
                last_activity = now
                last_signature = signature
            alerts = []
            terminal = state.get('state') in ('complete', 'failed', 'needs_analysis', 'paused')
            if not workflow_alive and not terminal:
                missing_since = missing_since or now
                if now - missing_since > 60:
                    alerts.append('Workflow process missing while status is nonterminal')
            else:
                missing_since = None
            if state.get('state') in ('failed', 'needs_analysis'):
                alerts.append('Workflow stopped: ' + state.get('detail', ''))
            if not terminal and now - last_activity > stall_seconds:
                alerts.append(f'No process CPU or progress activity for {stall_seconds} seconds')
            scratch = Path('/tmp/jiangzhikun_self_forcing_sparse_v02')
            free = shutil.disk_usage(scratch if scratch.exists() else HOME).free
            if free < 5_000_000_000:
                alerts.append('Experiment storage has less than 5 GB free')
            record = dict(checked_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                          monitor_pid=os.getpid(), workflow_pid=workflow_pid,
                          workflow_alive=workflow_alive, status=state, processes=active,
                          progress=progress, storage_free_bytes=free, alerts=alerts,
                          idle_seconds=round(now - last_activity, 1))
            protocol = read_json(HOME / 'protocol.json') or {}
            directories = [HOME / 'runs/full' / variant / f'seed{seed}'
                           for variant in protocol.get('variants', [])
                           for seed in protocol.get('router_seeds', [])]
            record['overall'] = dict(
                full_training_done=sum((d / 'training_completed.json').exists() for d in directories),
                full_training_total=len(directories),
                full_evaluation_done=sum((d / f'eval_{s}' / 'summary.json').exists()
                                         for d in directories for s in ('train', 'val', 'test')),
                full_evaluation_total=3 * len(directories))
            write_json(HOME / 'monitor.json', record)
            text = ['# 运行监控', '', record['checked_utc'] + ' UTC', '',
                    f'工作流存活：{workflow_alive}；监控 PID：{os.getpid()}。', '',
                    '阶段：' + state.get('phase', 'unknown'), '', state.get('detail', ''), '']
            overall = record['overall']
            text.extend([f"Full: training {overall['full_training_done']}/{overall['full_training_total']}; "
                         f"evaluations {overall['full_evaluation_done']}/{overall['full_evaluation_total']}", ''])
            for item in progress:
                text.extend([json.dumps(item, ensure_ascii=False), ''])
            text.extend(['告警：' + ('；'.join(alerts) if alerts else '无'), '',
                         '每 30 秒读取进程、CPU 活动和进度；不更改配置、不重启或终止任务。', ''])
            target = HOME / 'MONITOR.md'
            if target.is_symlink():
                target = target.resolve()
            temporary = target.with_suffix('.md.tmp')
            temporary.write_text('\n'.join(text))
            temporary.replace(target)
            notice = json.dumps([state, progress, alerts, workflow_alive], sort_keys=True)
            if notice != last_notice:
                with (logs / 'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(record, ensure_ascii=False), flush=True)
                last_notice = notice
            # Keep observing failed runs so a later supervised recovery is
            # noticed. Only completion or an explicit pause ends monitoring.
            if state.get('state') in ('complete', 'paused') and not workflow_alive:
                return
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--stall-seconds', type=int, default=900)
    args = parser.parse_args()
    if args.interval <= 0 or args.stall_seconds <= 0:
        parser.error('Intervals must be positive')
    run(args.interval, args.stall_seconds)
