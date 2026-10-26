"""Read-only Chinese dashboard; no training or GPU dependencies."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
STAGES = {'audit': '正确性检查', 'benchmark': '速度测量', 'collection': '数据采集',
          'training': 'Router 训练', 'assembling': '整理权重',
          'network_quality': '整网与视频验证', 'all_finished': '全部完成'}
STATES = {'training': '训练中', 'validating': '验证中', 'initializing': '初始化',
          'preflight_passed': '准备训练', 'completed': '已完成', 'paused': '已暂停',
          'failed': '发生错误', 'loading': '载入中'}


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '等待计时'
    minutes = max(1, math.ceil(seconds / 60))
    if minutes < 60:
        return f'{minutes} 分钟'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} 小时' + (f' {minutes} 分钟' if minutes else '')


def bar(done, total, width=24):
    ratio = min(1., max(0., done / max(total, 1)))
    filled = int(ratio * width)
    return '█' * filled + '░' * (width - filled)


def cells_width(value):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(value))


def table(headers, rows):
    rows = [headers, *rows]
    widths = [max(cells_width(r[i]) for r in rows) + 3 for i in range(len(headers))]
    return ['  ' + ''.join(str(v) + ' ' * (w - cells_width(v)) for v, w in zip(row, widths)).rstrip()
            for row in rows]


def train_eta(workers, manifest):
    estimates = []
    for worker in workers:
        active = [s for s in worker.get('layer_summaries', {}).values() if not s['done']]
        if not active or worker.get('status') == 'completed':
            continue
        speed = worker.get('seconds_mean')
        if not speed:
            return None
        remaining = max(manifest['max_updates'] - s['updates'] for s in active)
        validation = sum((s.get('curve', {}).get('K') or [{}])[-1].get('paired_validation_seconds', 0)
                         for s in active)
        estimates.append(remaining * speed + math.ceil(remaining / manifest['validate_every']) * validation)
    return max(estimates) if estimates else 0


def local_metrics(cells):
    cells = [s for s in cells.values() if all(s.get('curve', {}).get(k) for k in ('K', 'V'))]
    if not cells:
        return [], 0
    avg = lambda values: sum(values) / len(values)
    baseline = avg([s['curve']['K'][0]['prompt_equal_nmse'] for s in cells])
    rows = [['共同起点', f'{baseline:.8f}', '—']]
    for key, name in [('K', 'K-only'), ('V', 'V-aware')]:
        value = avg([s['curve'][key][-1]['prompt_equal_nmse'] for s in cells])
        improvement = (baseline - value) / max(baseline, 1e-12) * 100
        change = f'降低 {improvement:.2f}%' if improvement >= 0 else f'升高 {-improvement:.2f}%'
        rows.append([name, f'{value:.8f}', change])
    return rows, len(cells)


def show(details=False, color=False, watch=False):
    pointer = read(ROOT / 'migration/local_router_active.json')
    if not pointer.get('directory'):
        return '尚未找到当前 Router 实验。'
    root = Path(pointer['directory'])
    manifest = read(root / 'experiment.json')
    state = read(root / 'status.json')
    if not manifest:
        return f'实验配置暂不可读：{root}'
    workers = [read(root / f'training/rank{i}.json') for i in range(8)]
    workers = [w for w in workers if w]
    cells = {int(layer): summary for w in workers for layer, summary in w.get('layer_summaries', {}).items()}
    total = len(manifest['layers'])
    finished = sum(s['done'] for s in cells.values())
    first_round = sum(s['updates'] >= manifest['min_updates'] for s in cells.values())
    stage, status = state.get('stage'), state.get('status')
    tint = lambda text, code: f'\033[{code}m{text}\033[0m' if color else text
    now = datetime.now().astimezone().strftime('%H:%M:%S')
    lines = [tint(f'Router 实验进度  ·  {now}', '1;36'), '─' * 62]
    if status == 'failed':
        lines += [tint(f'发生错误：{STAGES.get(stage, stage)}', '1;31'), str(state.get('error', '请用 --details 查看日志目录'))]
    elif status == 'paused':
        lines.append(tint(f'已暂停  ·  {STAGES.get(stage, stage)}', '1;33'))
    elif status == 'completed':
        lines.append(tint('全部完成：训练、整网误差和视频验证均已结束', '1;32'))
    else:
        lines.append(tint(f'正在进行：{STAGES.get(stage, stage or "准备中")}', '1;32'))
    collection_done = len(list((Path(manifest['cache']) / 'receipts').glob('*.json')))
    event_total = len(manifest['rows'])
    collected_label = '采集完成' if collection_done == event_total else '采集中' if stage == 'collection' else '等待采集'
    training_label = '训练完成' if finished == total else '训练中' if stage == 'training' else '等待训练'
    quality_label = '验证完成' if status == 'completed' else '整网/视频验证中' if stage == 'network_quality' else '待验证视频'
    lines += [f'  {collected_label}  →  {training_label}  →  {quality_label}', '']
    if stage in ('audit', 'benchmark', 'collection'):
        lines += [f'数据采集  [{bar(collection_done, event_total)}]  {collection_done}/{event_total} 个事件',
                  f'覆盖 {total} 层 · 16 个训练 / 4 个验证 Prompt · 各 2 个生成 Seed']
        times = [read(root / f'collection/rank{i}.json') for i in range(8)]
        estimates = [(v['total'] - v['completed']) * v['seconds_mean'] for v in times
                     if v.get('seconds_mean') and v.get('status') != 'completed']
        if estimates:
            lines.append(f'采集剩余约 {duration(max(estimates))}')
    else:
        lines += [f'已完成层  [{bar(finished, total)}]  {finished}/{total} 层  ({finished / max(total, 1):.0%})']
        if stage == 'training':
            continuing = [s for s in cells.values() if not s['done']]
            active_workers = sum(w.get('status') != 'completed' for w in workers)
            lines.append(f'首轮目标  {first_round}/{total} 层已达到 {manifest["min_updates"]} 次更新')
            if continuing:
                low, high = min(s['updates'] for s in continuing), max(s['updates'] for s in continuing)
                progress = str(low) if low == high else f'{low}–{high}'
                lines.append(f'继续优化  {len(continuing)} 层 · 当前 {progress} 次更新 · 上限 {manifest["max_updates"]} 次')
            lines.append(f'运行情况  {active_workers}/8 张卡仍在工作 · K-only / V-aware 配对训练')
            eta = train_eta(workers, manifest)
            if eta:
                lines.append(tint(f'训练剩余  约 {duration(eta)}（按更新上限估计，可能提前结束）', '1;33'))
            elif eta is None:
                lines.append('训练剩余  正在初始化，完成首批更新后估算')
            if workers:
                lines.append(f'训练已用  {duration(max(w.get("elapsed_seconds", 0) for w in workers))}；视频验证耗时另计')
        rows, coverage = local_metrics(cells)
        if rows:
            lines += ['', tint(f'最近局部验证  ·  {coverage}/{total} 层等权平均  ·  误差越低越好', '1')]
            lines += table(['方案', 'Attention NMSE', '相对共同起点'], rows)
            lines.append('  这是冻结 Dense 输入下的局部误差，视频效果以整网验证为准。')
    if stage == 'network_quality':
        quality = root / 'network_check/dense0/quality'
        runs = [read(quality / f'rank{i}/status.json') for i in range(8)]
        videos = sum(v.get('completed', 0) for v in runs)
        lines += ['', f'实际视频验证  [{bar(videos, 16)}]  {videos}/16 组 Prompt / Seed']
        lines.append(f'  {sum(v.get("status") == "offline" for v in runs)} 卡计算整网误差 · '
                     f'{sum(v.get("status") == "rollout" for v in runs)} 卡生成视频')
    if status == 'completed' and state.get('report'):
        lines += ['', f'结果报告：{state["report"]}']
    if details:
        lines += ['', tint('逐层详情（第 0 层保持 Dense）', '1')]
        rows = []
        for layer in sorted(cells):
            s = cells[layer]
            latest = lambda k: f'{s["curve"][k][-1]["prompt_equal_nmse"]:.6f}' if s.get('curve', {}).get(k) else '等待验证'
            rows.append([layer, '已结束' if s['done'] else '优化中', s['updates'], latest('K'), latest('V')])
        if rows:
            lines += table(['层', '状态', '更新次数', 'K-only NMSE', 'V-aware NMSE'], rows)
        lines += ['', '逐卡状态']
        lines += table(['GPU', '负责层', '当前状态', '秒/更新'], [
            [w['rank'], ','.join(map(str, w['layers'])), STATES.get(w.get('status'), w.get('status')),
             f'{w["seconds_mean"]:.2f}' if w.get('seconds_mean') else '—'] for w in workers])
        lines += ['', f'实验目录：{root}', f'日志目录：{root / ("quality_workers" if stage == "network_quality" else "training_workers")}']
    footer = '每 5 秒刷新 · Ctrl+C 退出查看，训练继续' if watch else '加 --watch 自动刷新'
    lines += ['', footer + ' · 加 --details 查看逐层详情']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description='Router 实验中文进度总览')
    p.add_argument('--watch', action='store_true', help='每 5 秒自动刷新')
    p.add_argument('--details', action='store_true', help='显示逐层指标和逐卡状态')
    p.add_argument('--no-color', action='store_true', help='禁用终端颜色')
    args = p.parse_args()
    interactive = sys.stdout.isatty()
    alternate = args.watch and interactive
    try:
        if alternate:
            print('\033[?1049h\033[?25l', end='', flush=True)
        while True:
            view = show(details=args.details, color=interactive and not args.no_color, watch=args.watch)
            print(('\033[H' if alternate else '') + view + ('\033[J' if alternate else ''), flush=True)
            if not args.watch:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        if alternate:
            print('\033[?25h\033[?1049l', end='', flush=True)


if __name__ == '__main__':
    main()
