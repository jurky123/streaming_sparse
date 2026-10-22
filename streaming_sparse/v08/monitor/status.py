"""Chinese terminal dashboard. Reads metadata only; never launches workers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
GROUPS = ('F-K', 'F-V', 'A-K', 'A-V', 'J-K', 'J-V')
LABELS = {'prepare': '数据与审计', 'pilot': '校准与小样本检查',
          'matrix': '正式六组×三种子', 'final': '锁定与验证集评估', 'test': '独立测试与闭环'}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def source_sha():
    # Match common.source_identity without importing torch or loading any models.
    paths = list(HERE.glob('*.py')) + [HERE / 'configs/protocol.json']
    paths += [ROOT / p for p in ('utils/wan_wrapper.py', 'utils/scheduler.py',
        'wan/modules/causal_model.py', 'wan/modules/model.py', 'wan/modules/attention.py',
        'streaming_sparse/v05/primitives.py', 'streaming_sparse/capture/capture_runner.py',
        'configs/default_config.yaml', 'configs/self_forcing_dmd.yaml',
        'wan_models/Wan2.1-T2V-1.3B/config.json')]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def age(path):
    try:
        seconds = max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return '无记录'
    if seconds < 60:
        return f'{seconds}秒前'
    if seconds < 3600:
        return f'{seconds // 60}分钟前'
    return f'{seconds / 3600:.1f}小时前'


def bar(done, total, width=18):
    fraction = min(1, max(0, done / total)) if total else 0
    filled = int(width * fraction)
    return '[' + '#' * filled + '.' * (width - filled) + f'] {done}/{total} ({fraction:.1%})'


def alive(entry):
    """Verify the saved PID still belongs to a v08 worker, not just PID existence."""
    pid = entry.get('pid')
    if not isinstance(pid, int):
        return None
    try:
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        return 'streaming_sparse.v08.' in cmdline
    except FileNotFoundError:
        return False
    except OSError:
        return None


def job_directory(name, definition):
    match = re.fullmatch(r'(calibration|heads_[KV]|(?:pilot|main)_[FAJ]-[KV])_seed(\d+)', name)
    if match:
        kind, seed = match.groups()
        if kind == 'calibration':
            return HERE / f'runs/seed{seed}/calibration'
        if kind.startswith('heads_'):
            return HERE / f'runs/seed{seed}/warmstart_{kind[-1]}'
        stage, group = kind.split('_', 1)
        return HERE / f'{"diagnostics" if stage == "pilot" else "runs"}/seed{seed}/{group}'
    args = definition.get('args', [])
    if '--output' in args:
        return Path(args[args.index('--output') + 1])
    return None


def run_state(directory, entry, digest):
    complete = read(directory / 'completed.json')
    status = training_progress(directory)
    if complete:
        label = '完成' if complete.get('source_sha256') == digest else '旧版完成'
        return label, complete
    if entry.get('status') == 'failed':
        return '失败', status
    if entry.get('status') == 'running':
        live = alive(entry)
        label = '运行' if live else ('进程已退出?' if live is False else '进程待核实')
        return label, status
    if status:
        return '有进度/未确认运行', status
    return '排队' if entry.get('status') == 'pending' else '未启动', status


def training_progress(directory):
    """Status is written at validation; metrics records every finished update."""
    status = read(directory / 'status.json')
    metrics = directory / 'metrics.jsonl'
    try:
        with metrics.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read().splitlines()
        for line in reversed(tail):
            try:
                latest = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if latest.get('updates', -1) >= status.get('updates', 0):
                status = {**status, 'updates': latest['updates'], '_progress_file': str(metrics)}
            break
    except OSError:
        pass
    return status


def gpu_lines():
    try:
        result = subprocess.run(['nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if result.returncode:
            return ['  GPU 信息暂不可读（请在主机终端运行面板）']
        lines = []
        for fields in csv.reader(result.stdout.splitlines()):
            if len(fields) == 5:
                index, name, used, total, utilization = (s.strip() for s in fields)
                lines.append(f'  GPU {index} | {name} | 显存 {int(used)/1024:.1f}/{int(total)/1024:.1f} GiB | 利用率 {utilization}%')
        return lines or ['  未返回 GPU 信息']
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return ['  GPU 信息暂不可读（请在主机终端运行面板）']


def render(no_gpu=False):
    digest = source_sha()
    plan = read(HERE / 'reports/plan.json')
    definitions = {j['name']: j for j in plan.get('jobs', [])}
    state_path = HERE / 'reports/workflow/status.json'
    workflow = read(state_path)
    jobs = workflow.get('jobs', {})
    counts = Counter(j.get('status', 'unknown') for j in jobs.values())
    active = [(n, j) for n, j in jobs.items() if j.get('status') == 'running']
    video_check=read(HERE/'reports/two_video_evaluation.json')
    evaluating=bool(video_check.get('training_temporarily_suspended')and video_check.get('status','').startswith('running'))
    lines = ['═' * 64, 'CAVIR v0.8  实验状态',
             datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S 北京时间'), '═' * 64]
    if evaluating:
        headline=f'两视频端到端评估中（{video_check.get("completed",0)}/2）；训练临时挂起，评估结束自动恢复'
    elif workflow.get('paused'):
        headline = '已按用户要求暂停；调度器已停止，等待项不会自动执行'
    elif workflow.get('failure') or counts['failed']:
        headline = '任务失败，后续派发已停止；查看下方日志'
    elif active:
        headline = f'队列记录 {len(active)} 个运行任务（下方核实进程）'
    elif jobs and counts['completed'] == len(jobs):
        headline = f'本次队列已完成，范围截至 {LABELS.get(workflow.get("through"), workflow.get("through"))}'
    elif jobs:
        headline = '有队列记录，当前无运行任务；调度器是否在线需核实'
    else:
        headline = '尚未启动 v0.8 实验队列；已完成实现与 Phase0 审计'
    lines += ['当前：' + headline, '']
    if evaluating:
        lines.append('  视频评估：'+str(video_check.get('status'))+' | '+str(video_check.get('output')))
    if workflow.get('execution_sha256'):
        lines.append('  执行：已审计加速版 | 每10步保存 recovery.pt')
        if workflow.get('execution_note'):
            lines.append('  '+str(workflow['execution_note']))
    if workflow.get('allowed_gpus') is not None:
        devices=', '.join(str(i)for i in workflow['allowed_gpus'])
        lines.append(f'  资源限制：仅物理 GPU {devices} | 最多 {workflow.get("gpu_worker_limit",len(workflow["allowed_gpus"]))} 个 GPU 任务')
    audits_ok = True
    for name, label in [('phase0_unit', '数学/小型网络'), ('phase0_real', '真实 1.3B 网络')]:
        audit = read(HERE / f'reports/{name}.json')
        ok = audit.get('passed') and audit.get('source_sha256') == digest
        audits_ok = audits_ok and bool(ok)
        result = '通过 ✓' if ok else ('源码变化，需重审' if audit.get('passed') else '未通过/无报告')
        lines.append(f'  Phase0 {label}：{result}')
    lines += [f'  科研源码：{digest[:16]}', '', '数据准备（完整 Replay 输入，非旧 QKV 样本）']
    data = {s: {} for s in ('train', 'val', 'test')}
    malformed = 0
    manifest = HERE / 'data/replay/events.jsonl'
    try:
        with manifest.open() as stream:
            for row in stream:
                if not row.strip():
                    continue
                try:
                    record = json.loads(row)
                    data[record['split']][record['sample_id']] = record
                except (ValueError, KeyError, TypeError):
                    malformed += 1
    except FileNotFoundError:
        pass
    for split, label, total in [('train', 'Train', 768), ('val', 'Val  ', 192), ('test', 'Test ', 384)]:
        rows = list(data[split].values())
        prompts = len({r['prompt_id'] for r in rows})
        trajectories = len({r['trajectory_id'] for r in rows})
        sealed = ' | 封存：锁定 checkpoint 后才采集/评估' if split == 'test' and not (HERE / 'reports/final_lock.json').exists() else ''
        lines.append(f'  {label} {bar(len(rows), total)} | {prompts} prompts / {trajectories} 条轨迹{sealed}')
    if malformed:
        lines.append(f'  注意：{malformed} 条清单记录不完整，可能正在写入或需要检查')
    lines += ['', '阶段队列（未纳入本次队列的阶段仅为计划）']
    for stage, label in LABELS.items():
        names = [n for n, j in definitions.items() if j['stage'] == stage]
        included = [n for n in names if n in jobs]
        if included:
            c = Counter(jobs[n]['status'] for n in included)
            detail = f'完成 {c["completed"]}/{len(included)}，运行 {c["running"]}，等待 {c["pending"]}，失败 {c["failed"]}'
        else:
            detail = '未启动（计划）'
        lines.append(f'  {label}：{detail}')
    if jobs:
        lines.append(f'  调度记录更新于 {age(state_path)}；任务节点包含采集/审计/评估，不等于训练组数')
        if workflow.get('source_sha256') != digest:
            lines.append('  注意：队列源码身份与当前源码不一致')
    lines += ['', '正式主实验：6 组 × 3 Router seeds（不计临时 pilot）',
              '  F=只训练 Router，A=只训练 LoRA，J=共同训练；K=QK-only，V=V-aware']
    completed = 0
    for group in GROUPS:
        cells = []
        for seed in range(3):
            label, info = run_state(HERE / f'runs/seed{seed}/{group}', jobs.get(f'main_{group}_seed{seed}', {}), digest)
            completed += label == '完成'
            updates = info.get('updates')
            suffix = f' {updates}/{info.get("total", 2000)}' if updates is not None else ''
            cells.append(f'seed{seed}: {label}{suffix}')
        lines.append('  ' + group + ' | ' + ' | '.join(cells))
    lines += [f'  正式训练已完成 {completed}/18 组', '', '当前任务 / 异常任务']
    selected = [(n, j) for n, j in jobs.items() if j.get('status') in ('running', 'failed')]
    if not selected:
        lines.append('  无运行或失败任务记录')
    for name, entry in selected:
        live = alive(entry) if entry.get('status') == 'running' else None
        label = '失败' if entry.get('status') == 'failed' else ('运行中' if live else ('PID 已退出/不可见，勿当作仍在运行' if live is False else '进程待核实'))
        if evaluating and entry.get('pid')in video_check.get('paused_pids',[]):
            label='临时挂起（两视频评估，保留内存进度）'
        lines.append(f'  {name} | {label} | GPU {entry.get("gpu")} | PID {entry.get("pid")}')
        directory = job_directory(name, definitions.get(name, {}))
        if directory:
            info = training_progress(directory)
            if info:
                progress_file = Path(info.get('_progress_file', directory / 'status.json'))
                lines.append('    更新 ' + bar(info.get('updates', 0), info.get('total', info.get('max_updates', 2000))) + f' | 记录 {age(progress_file)}')
                if info.get('best') is not None:
                    lines.append(f'    Val-best {info["best"]} @ update {info.get("best_step", "?")}')
            progress = list(directory.glob('validation/*/progress.json')) + list(directory.glob('keep*/progress.json'))
            if (directory / 'progress.json').exists():
                progress.append(directory / 'progress.json')
            if progress:
                latest = max(progress, key=lambda p: p.stat().st_mtime)
                current = read(latest)
                if current:
                    lines.append(f'    评估/闭环 {bar(current.get("completed", 0), current.get("total", 0))} | {age(latest)}')
        if entry.get('log'):
            lines.append('    日志：' + entry['log'])
    if workflow.get('failure'):
        lines.append('  停止原因：' + str(workflow['failure']))
    if workflow.get('paused'):
        lines.append('  暂停原因：' + str(workflow.get('pause_reason', '用户要求暂停')))
    lines += ['', 'GPU（整卡占用可能包含其他实验）']
    lines += ['  已跳过 GPU 查询'] if no_gpu else gpu_lines()
    if not jobs:
        if audits_ok:
            lines += ['', '下一阶段：补齐 Train/Val → 共同校准 → K/V warm-start → seed0 四组 pilot']
        lines.append('  本面板只读状态，不启动、停止或恢复实验。')
    lines += ['', '刷新：status.sh --watch  （默认 5 秒；Ctrl+C 仅退出面板）']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', nargs='?', const=5.0, type=float, metavar='SECONDS', help='自动刷新，默认 5 秒')
    parser.add_argument('--no-gpu', action='store_true', help='仅查看磁盘记录，不查询 GPU')
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1:
        parser.error('--watch 间隔至少 1 秒')
    try:
        while True:
            if args.watch is not None and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(render(no_gpu=args.no_gpu), flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print('\n已退出状态面板；实验任务不受影响。')


if __name__ == '__main__':
    main()
