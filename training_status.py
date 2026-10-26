#!/usr/bin/env python3
"""Read-only terminal status for Opt30/31/32. Standard library only.
Never imports experiment reporters, writes summaries, or controls processes.
"""
import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time
import unicodedata

DEFAULT = Path(__file__).resolve().parent / 'streaming_sparse/v10/experiments'
ROOTS = {'Opt30': 'opt30_behavior_grouping_20261026',
         'Opt31': 'opt31_priority_long_video_20261026',
         'Opt32': 'opt32_actions_vaware_20261026'}
FINALISTS = ['q32_full', 'q64_mean8', 'q64_qk_mean16', 'q64_qk_local16', 'q32_qk_local16']
ACTIONS = ['Base', 'K32_analytic', 'V32_analytic', 'K32_action', 'V32_action',
           'V8_action', 'K32_dynamic', 'V32_dynamic']
STAGES = {'completed': '已完成', 'pilot_completed': '已完成（4层试验）',
          'final': '全层训练/测试中', 'pilot': '4层试验中', 'evaluation': '视频评测中',
          'waiting_priority_and_opt30': '等待 Opt31 评测与 Opt30 基础实验',
          'complete_with_failures': '结束，但有失败',
          'pilot_complete_with_failures': '4层试验结束，但有失败',
          'blocked_missing_complete_foundation': '阻断：基础权重未完整就绪'}
FAILURES = {'failed', 'blocked', 'orphaned'}


def read(path, issues):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        issues.append(f'读取异常 {path}: {exc}')
        return {}


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def age(path, now):
    try:
        return max(0, now - path.stat().st_mtime)
    except OSError:
        return None


def elapsed(seconds):
    if seconds is None:
        return '—'
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02}s'
    return f'{seconds // 3600}h{seconds % 3600 // 60:02}m'


def arg(job, key, default=None):
    args = job.get('args', [])
    try:
        return args[args.index(key) + 1]
    except (ValueError, IndexError):
        return default


def job_kind(job, label):
    ident, module = job.get('id', ''), job.get('module', '')
    if ident.startswith('bench_') or module == 'bench':
        return '性能测试'
    if '--prepare' in job.get('args', []) or module == 'prepare':
        return '校准/分组准备'
    if module == 'preflight':
        return '正确性检查'
    if module in ('evaluate', 'e2e') or ident.startswith('e2e_'):
        return '视频评测'
    if label == 'Opt32' and arg(job, '--variant') in ('Base', 'K32_analytic', 'V32_analytic'):
        return '解析对照评测'
    return '训练'


def snapshot(base):
    now = time.time()
    result = {'utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
              'queues': {}, 'routes': {}, 'active': [], 'alerts': [], 'notes': []}
    for label, dirname in ROOTS.items():
        root = base / dirname
        q = read(root / 'queue.json', result['alerts'])
        if not q:
            result['alerts'].append(f'{label} 无可读队列：{root / "queue.json"}')
        jobs = q.get('jobs', [])
        counts = dict(collections.Counter(j.get('status', 'unknown') for j in jobs))
        visible = [j for j in jobs if j.get('status') != 'superseded']
        pending = collections.Counter(job_kind(j, label) for j in visible if j.get('status') == 'pending')
        heartbeat = now - q.get('updated', now)
        result['queues'][label] = {'stage': q.get('stage', '无数据'), 'counts': counts,
                                  'total': len(visible), 'pending_types': dict(pending),
                                  'heartbeat_seconds': max(0, heartbeat)}
        if counts.get('running', 0) and heartbeat > 120:
            result['alerts'].append(f'{label} 队列心跳 {elapsed(heartbeat)} 未刷新；需核查调度器，不能据此断言训练停止')
        for j in visible:
            logfile = j.get('log') or str(root / 'logs' / (j.get('id', 'unknown') + '.log'))
            if j.get('status') in FAILURES:
                result['alerts'].append(f'{label} {j.get("id")} {j["status"]}：{j.get("reason", j.get("error", ""))} 日志 {logfile}')
            if j.get('status') != 'running':
                continue
            layer, variant = arg(j, '--layer'), arg(j, '--variant')
            folder = root / 'training' / f'layer{int(layer):02}' / variant if layer and variant else None
            status = read(folder / 'status.json', result['alerts']) if folder else {}
            best = read(folder / 'best_validation.json', result['alerts']) if folder else {}
            kind = job_kind(j, label)
            progress = '—'
            if kind == '训练':
                step, total = status.get('updates'), status.get('total')
                if step is not None and total:
                    progress = f'{step}/{total} ({100*step/total:.0f}%)'
                    if status.get('status') == 'completed':
                        progress = f'{step}/{total}；训练已结束'
                    elif step >= total:
                        progress += ' 收尾/验证'
                else:
                    progress = '加载/初始化'
            stamp = age(folder / 'status.json', now) if folder else None
            freshness = min(x for x in (stamp, age(Path(logfile), now)) if x is not None) if any(x is not None for x in (stamp, age(Path(logfile), now))) else None
            pid = j.get('pid')
            proc = Path(f'/proc/{pid}/stat')
            try:
                if pid and proc.read_text().split(') ', 1)[1].split()[0] == 'Z':
                    result['alerts'].append(f'{label} {j["id"]} 进程已退出，队列尚待刷新')
            except FileNotFoundError:
                if pid:
                    result['alerts'].append(f'{label} {j["id"]} PID {pid} 不在当前 /proc；可能刚结束或进程命名空间不同')
            except (OSError, IndexError):
                pass
            if freshness is not None and freshness > 600:
                result['alerts'].append(f'{label} {j["id"]} 状态/日志 {elapsed(freshness)} 无更新；验证或编译也可能暂不更新')
            result['active'].append({'gpu': j.get('gpu'), 'experiment': label, 'job': j.get('id'),
                                     'kind': kind, 'progress': progress,
                                     'best_nmse': best.get('nmse', status.get('best')),
                                     'elapsed': elapsed(now-j.get('started', now)),
                                     'last_update': elapsed(freshness), 'log': logfile})
        if label == 'Opt31':
            cases = [j for j in visible if j.get('module') == 'evaluate']
            result['queues'][label]['cases'] = [sum(j.get('status') == 'done' for j in cases), len(cases)]
            continue
        promoted = read(root / 'promotion.json', result['alerts']).get('selected', FINALISTS)
        variants, layers = (promoted, range(1, 30)) if label == 'Opt30' else (ACTIONS, [1, 14, 24, 29])
        routes = []
        for variant in variants:
            completed, bests, plateau, capped, infeasible = [], [], [], [], []
            for layer in layers:
                folder = root / 'training' / f'layer{layer:02}' / variant
                done = read(folder / 'completed.json', result['alerts'])
                if not done or done.get('completed') is False:
                    continue
                completed.append(layer)
                if done.get('convergence') == 'plateau':
                    plateau.append(layer)
                elif label == 'Opt30':
                    capped.append(layer)
                if label == 'Opt32' and not done.get('budget_feasible', False):
                    infeasible.append(layer)
                else:
                    value = done.get('best_native_nmse', done.get('best'))
                    if number(value):
                        bests.append(value)
            routes.append({'variant': variant, 'done': len(completed), 'target': len(layers),
                           'best_mean_nmse': sum(bests)/len(bests) if bests else None,
                           'metric_layers': len(bests), 'plateau': len(plateau),
                           'convergence_unconfirmed_layers': capped,
                           'budget_infeasible_layers': infeasible,
                           'remaining_layers': [x for x in layers if x not in completed]})
            if capped:
                result['notes'].append(f'{variant} L{",".join(map(str,capped))} 已结束，但未标记平台收敛；查看 curve.json，勿把跑完等同于收敛')
            if infeasible:
                result['notes'].append(f'{variant} L{",".join(map(str,infeasible))} 超出验证执行预算，不纳入可用 best 均值')
        result['routes'][label] = routes
    return result


def width(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(text))


def table(headers, rows):
    rows = [[str(x) for x in r] for r in [headers] + rows]
    widths = [max(width(r[i]) for r in rows) for i in range(len(headers))]
    return '\n'.join('  '.join(v + ' ' * (widths[i]-width(v)) for i, v in enumerate(r)).rstrip() for r in rows)


def fmt(value):
    return f'{value:.6f}' if number(value) else '—'


def render(s, details=False):
    lines = ['训练状态  |  ' + s['utc'], '只读快照：不写训练目录、不触发重试、不加载模型。', '']
    for label, q in s['queues'].items():
        c = q['counts'];done = c.get('done', 0);total = q['total'];filled = int(18*done/total) if total else 0
        lines.append(f'{label} [{"#"*filled}{"."*(18-filled)}] {done}/{total}  {STAGES.get(q["stage"],q["stage"])}')
        lines.append(f'      运行 {c.get("running",0)} | 排队 {c.get("pending",0)} | 失败/阻断 {sum(c.get(x,0) for x in FAILURES)}'
                     + (f' | 视频案例 {q["cases"][0]}/{q["cases"][1]}' if 'cases' in q else ''))
        if q['pending_types']:
            lines.append('      待运行：' + '，'.join(f'{k} {v}' for k,v in q['pending_types'].items()))
    lines += ['', 'GPU 当前任务（仅这三个实验；未列出的 GPU 不代表空闲）']
    active = sorted(s['active'], key=lambda r: (str(r['gpu']), r['job']))
    lines.append(table(['GPU', '实验/任务', '类型', '进度', 'Best NMSE', '已用时', '最近更新'],
                       [[r['gpu'], r['experiment']+'/'+r['job'], r['kind'], r['progress'], fmt(r['best_nmse']), r['elapsed'], r['last_update']] for r in active]) if active else '  无运行任务')
    for label, routes in s['routes'].items():
        lines += ['', label + (' 全层晋级路线（目标固定29层）' if label == 'Opt30' else ' 动作 / V-aware（目标固定4层；不是全层视频结果）')]
        lines.append(table(['路线', '完成/目标', 'Best NMSE均值', '指标层数', '收敛/预算'],
                           [[r['variant'], f'{r["done"]}/{r["target"]}', fmt(r['best_mean_nmse']), r['metric_layers'],
                             f'平台 {r["plateau"]}/{r["done"]}' if label == 'Opt30' else f'预算不合格 {len(r["budget_infeasible_layers"])}'] for r in routes]))
        if details:
            lines.extend(f'  {r["variant"]} 未完成层：{r["remaining_layers"]}' for r in routes if r['remaining_layers'])
    lines += ['', '异常：'] + (['  '+x for x in s['alerts']] or ['  未发现队列失败、阻断或明显状态异常'])
    lines += ['', '注意：部分层均值不能直接跨路线排名；平台早停是训练脚本判据，步数跑满不等于收敛。']
    lines += ['  '+x for x in s['notes']]
    if details:
        lines += ['', '运行任务日志（可用 tail -f 路径）：'] + [f'  {r["job"]}: {r["log"]}' for r in active]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch', nargs='?', type=float, const=10, metavar='SECONDS', help='持续刷新，默认10秒；Ctrl+C退出')
    p.add_argument('--details', action='store_true', help='列出未完成层和运行任务日志')
    p.add_argument('--json', action='store_true', help='仅向标准输出打印JSON；watch时每行一份快照')
    p.add_argument('--experiments', type=Path, default=DEFAULT, help='实验根目录，默认自动定位')
    a = p.parse_args()
    if a.watch is not None and a.watch < 1:
        p.error('--watch 间隔必须至少1秒')
    try:
        while True:
            s = snapshot(a.experiments)
            if a.watch and sys.stdout.isatty() and not a.json:
                print('\033[2J\033[H', end='')
            print(json.dumps(s, ensure_ascii=False) if a.json else render(s, a.details), flush=True)
            if a.watch is None:
                break
            time.sleep(a.watch)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
