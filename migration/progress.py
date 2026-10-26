#!/usr/bin/env python3
"""Read-only terminal dashboard for the active eight-GPU continuation."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import shutil
import re
import statistics
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'streaming_sparse/v08/runs/seed0/F-K'
DEFAULT_STATE = ROOT / 'migration/restoration/20261022_133350'


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default if default is not None else {}


def read_metrics(path):
    rows = {}
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if isinstance(row.get('updates'), int):
                        rows[row['updates']] = row
                except ValueError:
                    pass  # A writer may still be appending the final line.
    except OSError:
        pass
    return [rows[step] for step in sorted(rows)]


def age(path, now):
    try:
        return max(0, now - path.stat().st_mtime)
    except OSError:
        return None


def alive(pid):
    try:
        process = Path('/proc') / str(int(pid))
        state = next(line for line in (process / 'status').read_text().splitlines() if line.startswith('State:'))
        command = (process / 'cmdline').read_bytes()
        return 'Z' not in state and b'migration.runtime' in command
    except (OSError, ValueError, TypeError, StopIteration):
        return False


def stamp(value):
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def number(value, fallback='—'):
    return f'{value:.4e}' if isinstance(value, (int, float)) and math.isfinite(value) else fallback


def duration(seconds):
    if seconds is None:
        return '—'
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours}小时{minutes:02d}分'
    if minutes:
        return f'{minutes}分{seconds:02d}秒'
    return f'{seconds}秒'


def sparkline(values):
    if not values:
        return '—'
    lo, hi = min(values), max(values)
    if hi == lo:
        return '▄' * len(values)
    bars = '▁▂▃▄▅▆▇█'
    return ''.join(bars[min(7, int(7 * (value-lo) / (hi-lo)))] for value in values)


def gpu_info():
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        rows = []
        for row in csv.reader(io.StringIO(result.stdout)):
            def numeric(value):
                try: return float(value.strip())
                except ValueError: return None
            rows.append(dict(zip(('index', 'util', 'used', 'total', 'temp'), map(numeric, row))))
        return rows, None
    except (OSError, subprocess.SubprocessError) as error:
        return [], f'GPU 状态暂不可用：{type(error).__name__}'


def validation_progress(run, step, curve):
    if any(row.get('updates') == step for row in curve):
        return None
    directory = run / f'validation/update{step:06d}'
    paths = list(directory.glob('rank*/progress.json'))
    if not paths and not (directory / 'progress.json').exists():
        return None
    paths = paths or [directory / 'progress.json']
    progress = [read_json(path) for path in paths]
    return dict(completed=sum(row.get('completed', 0) for row in progress),
                total=192, paths=paths)


def last_error(state_dir):
    for path in sorted((state_dir / 'batch8_training').glob('rank*.log')):
        try:
            with path.open('rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-16384))
                lines = stream.read().decode(errors='replace').splitlines()
            for line in reversed(lines):
                if 'Error:' in line or 'Exception:' in line:
                    return path.name + ': ' + line.removeprefix('[rank0]: ')
        except OSError:
            pass
    return None


def collect(run, state_dir, include_gpu=True):
    now = time.time()
    metrics = read_metrics(run / 'metrics.jsonl')
    live = read_json(state_dir / 'continuation_status.json')
    launch = read_json(state_dir / 'batch8_training/launcher_status.json')
    recovery = read_json(run / 'recovery.json')
    parallel = read_json(run / 'parallel_status.json')
    provenance = read_json(run / 'provenance.json')
    settings = provenance.get('settings', {})
    total = settings.get('updates', 2000)
    batch = live.get('global_batch', parallel.get('global_batch', 8))
    step = max([0, recovery.get('updates', 0), parallel.get('updates', 0)] +
               [row['updates'] for row in metrics[-1:]])
    recent = metrics[-20:]
    # Show speed for the current execution version immediately after a switch.
    # Older losses remain visible, but their timing should not bias the ETA.
    current_audit=recent[-1].get('routerslim',{}).get('audit_sha256')if recent else None
    speed_rows=[row for row in recent if row.get('routerslim',{}).get('audit_sha256')==current_audit]
    times = [row.get('execution_timing', {}).get('update_seconds') for row in speed_rows]
    times = [value for value in times if isinstance(value, (int, float)) and math.isfinite(value) and value > 0]
    seconds = statistics.median(times) if times else None
    curve = read_json(run / 'validation_curve.json', [])
    completed = (run / 'completed.json').exists() and step >= total
    ranks = launch.get('ranks', [])
    active = sum(alive(row.get('pid')) for row in ranks)
    world = launch.get('world_size', parallel.get('world_size', 8))
    progress = validation_progress(run, step, curve)
    changed_age = age(run / 'metrics.jsonl', now)
    if progress:
        available = [age(path, now) for path in progress['paths']]
        changed_age = min((value for value in available if value is not None), default=changed_age)
    if completed:
        status, level = '已完成', 'ok'
    elif live.get('status')=='paused'and not active:
        status,level='已暂停','warn'
    elif launch.get('status') == 'failed' or live.get('status') == 'multigpu_failed':
        status, level = '训练失败', 'error'
    elif not active:
        investigation=read_json(ROOT/'migration/routerslim_active.json')
        if investigation.get('status')in ('profiling_preparation','profiling','auditing'):
            status,level=f"R1 profile / 等价性检查（训练暂停于 {step}）",'warn'
        elif investigation.get('status')=='resuming':
            status,level='R1 验收通过，正在恢复训练','warn'
        else:
            status, level = ('已停止', 'error') if ranks else ('尚未启动', 'warn')
    elif active < world:
        status, level = f'进程不完整（{active}/{world}）', 'error'
    elif progress:
        status, level = f"验证中 {progress['completed']}/{progress['total']}", 'ok'
    elif live.get('status') == 'starting_multigpu':
        status, level = '加载模型', 'warn'
    elif changed_age is not None and changed_age > max(120, 5 * (seconds or 25)):
        status, level = '等待新进度', 'warn'
    else:
        status, level = '训练中', 'ok'
    gpus, gpu_error = gpu_info() if include_gpu else ([], None)
    selected = {int(value) for value in launch.get('gpus', list(range(world)))}
    gpus = [row for row in gpus if row['index'] is not None and int(row['index']) in selected]
    valid_scores = [row for row in curve if isinstance(row.get('prompt_equal_error'), (int, float))]
    best = min(valid_scores, key=lambda row: row['prompt_equal_error']) if valid_scores else None
    last_val = max((row.get('updates', 0) for row in curve), default=None)
    remaining = max(0, total-step) * seconds if seconds else None
    return dict(now=now, run=run, state_dir=state_dir, step=step, total=total, batch=batch,
        status=status, level=level, active=active, world=world, recent=recent, seconds=seconds,
        remaining=remaining, curve=curve, best=best, last_val=last_val,speed_samples=len(times),
        next_val=min(total, (step // settings.get('validate_every', 200)+1)*settings.get('validate_every', 200)),
        checkpoint=recovery.get('updates'), checkpoint_age=age(run / 'recovery.pt', now),
        changed_age=changed_age, elapsed=now-stamp(launch['started_at']) if stamp(launch.get('started_at')) else None,
        error=last_error(state_dir) if level == 'error' and not completed else None,
        gpus=gpus, gpu_error=gpu_error)


def fit(text, width):
    plain = re.sub(r'\x1b\[[0-9;]*m', '', text)
    cells = sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in 'WF' else 1 for char in plain)
    if cells <= width:
        return text
    result, cells = '', 0
    for token in re.split(r'(\x1b\[[0-9;]*m)', text):
        if token.startswith('\x1b['):
            result += token
            continue
        for character in token:
            size = 0 if unicodedata.combining(character) else 2 if unicodedata.east_asian_width(character) in 'WF' else 1
            if cells + size > width-1:
                return result + '…' + ('\033[0m' if '\033[' in text else '')
            result += character
            cells += size
    return result


def render(data, color=False, width=88, height=40):
    line = '─' * min(width, 88)
    def paint(text, code): return f'\033[{code}m{text}\033[0m' if color else text
    status = paint(data['status'], {'ok': '32', 'warn': '33', 'error': '31'}[data['level']])
    step, total = data['step'], data['total']
    ratio = min(1, step/max(1, total))
    bar_width = max(8, min(32, width-32))
    units = ratio*bar_width
    filled = int(units)
    fraction = int((units-filled)*8)
    partial = '▏▎▍▌▋▊▉'[fraction-1] if fraction and filled < bar_width else ''
    bar = '█'*filled + partial + '░'*(bar_width-filled-bool(partial))
    current = datetime.fromtimestamp(data['now'], timezone.utc).strftime('%m-%d %H:%M:%S UTC')
    out = [f"CAVIR 训练进度  ·  {data['run'].name} / {data['run'].parent.name}     {current}", line,
        f"{status}   |   进程 {data['active']}/{data['world']}   |   Batch {data['batch']}   |   最近更新 {duration(data['changed_age'])}前",
        f'[{bar}]  {step:,} / {total:,}  ({100*ratio:.2f}%)']
    seconds = data['seconds']
    out.append(f"近{data['speed_samples']}步同配置中位耗时  {seconds:.1f} 秒/步   ·   吞吐 {data['batch']/seconds:.2f} 输入/秒" if seconds else '速度：等待完成首个更新')
    eta = datetime.fromtimestamp(data['now']+data['remaining'], timezone.utc).strftime('%m-%d %H:%M UTC') if data['remaining'] is not None else '—'
    out.append(f"预计剩余 ≈{duration(data['remaining'])}   ·   预计结束 {eta}")
    out.append(f"样本 {step*data['batch']:,} / {total*data['batch']:,}   ·   本次运行 {duration(data['elapsed'])}   ·   恢复点 step {data['checkpoint'] if data['checkpoint'] is not None else '—'}")
    out.append('（预计时间按近期训练速度推算，不含验证与停机）')
    out.extend(['', 'LOSS / 验证'])
    recent = data['recent']
    averages = {}
    for name in ('output', 'att_aux', 'cal_aux'):
        values = [row.get('losses', {}).get(name) for row in recent]
        values = [value for value in values if isinstance(value, (float, int)) and math.isfinite(value)]
        averages[name] = statistics.mean(values) if values else None
    attn='停算（权重0）'if recent and recent[-1].get('att_aux_computed')is False else number(averages['att_aux'])
    out.append(f"近{len(recent)}步均值   Output {number(averages['output'])}   Attn {attn}   Cal {number(averages['cal_aux'])}")
    values = [row.get('losses', {}).get('output') for row in recent]
    values = [value for value in values if isinstance(value, (float, int)) and math.isfinite(value)]
    out.append(f"Output 趋势    {sparkline(values)}   （左→右为时间，低→高为loss）")
    best = data['best']
    out.append(f"验证最佳 NMSE  {number(best['prompt_equal_error'])} @ step {best['updates']}" if best else '验证：等待首次结果')
    out[-1] += f"   ·   下次 step {data['next_val']}"
    out.extend(['', 'GPU    占用        显存 GiB         温度' if height >= 28 else 'GPU 占用 / 显存 GiB'])
    gpu_lines = []
    for gpu in data['gpus']:
        util = f"{gpu['util']:3.0f}%" if gpu['util'] is not None else '  — '
        memory = f"{gpu['used']/1024:5.1f} / {gpu['total']/1024:4.1f}" if gpu['used'] is not None and gpu['total'] else '       —      '
        temperature = f"{gpu['temp']:.0f}°C" if gpu['temp'] is not None else '—'
        if height >= 28:
            gpu_lines.append(f" {int(gpu['index'])}     {util}       {memory}       {temperature}")
        else:
            gpu_lines.append(f"GPU{int(gpu['index'])} {util}  {memory.replace(' ', '')}G")
    if height < 28:
        middle = (len(gpu_lines)+1)//2
        for i in range(middle):
            out.append(gpu_lines[i].ljust(29) + (gpu_lines[i+middle] if i+middle < len(gpu_lines) else ''))
    else:
        out.extend(gpu_lines)
    if not data['gpus']: out.append(data['gpu_error'] or '未查询 GPU')
    if height >= 34:
        out.extend(['', '最近更新     秒/步       Output loss       Router 梯度'])
        for row in recent[-4:]:
            value = row.get('execution_timing', {}).get('update_seconds')
            took = f'{value:6.1f}' if isinstance(value, (float, int)) else '     —'
            gradient = row.get('gradient_norm_before_clip', {}).get('router')
            out.append(f"  {row['updates']:5d}     {took}       {number(row.get('losses', {}).get('output')):>10}        {number(gradient)}")
    if data['error']: out.extend(['', paint(data['error'][:max(60, width)], '31')])
    out.extend([line, '每5秒刷新 · Ctrl+C 退出 · --once 查看单次快照'])
    if len(out) > height:
        out = [text for text in out if text]
    if len(out) > height:
        out = [text for text in out if not text.startswith('Output 趋势')]
    return '\n'.join(fit(text, width) for text in out[:max(1, height)])


def main():
    parser = argparse.ArgumentParser(description='CAVIR 八卡训练进度面板（只读，无需安装依赖）')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='只显示一次')
    mode.add_argument('--watch', action='store_true', help='持续刷新；终端内默认启用')
    parser.add_argument('--interval', type=float, default=5, help='刷新间隔，默认5秒')
    parser.add_argument('--no-gpu', action='store_true', help='不查询nvidia-smi')
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    if args.interval < 1: parser.error('--interval 不能小于1秒')
    terminal = sys.stdout.isatty()
    watching = not args.once and (terminal or args.watch)
    color = terminal and not args.no_color and not os.environ.get('NO_COLOR')
    try:
        if terminal and watching: print('\033[?1049h\033[?25l', end='', flush=True)
        while True:
            data = collect(args.run_dir.resolve(), args.state_dir.resolve(), not args.no_gpu)
            size = shutil.get_terminal_size((88, 40))
            text = render(data, color, size.columns, size.lines)
            text = text.replace('每5秒刷新', f'每{args.interval:g}秒刷新') if watching else text.replace('每5秒刷新 · Ctrl+C 退出 · ', '')
            if terminal and watching: print('\033[H\033[2J', end='')
            print(text, flush=True)
            if not watching: break
            time.sleep(args.interval)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        if terminal and watching: print('\033[0m\033[?25h\033[?1049l', end='', flush=True)


if __name__ == '__main__':
    main()
