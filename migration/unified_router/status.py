"""Chinese, read-only terminal dashboard. No model or training imports."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
GROUP_NAMES = {'K32-Set': 'K32 集合', 'V8-Set': 'V8 集合',
               'V32-NoSet': 'V32 无集合', 'V32-Set': 'V32 集合'}
STAGES = {'P0_math': '数学与边界检查', 'K_initialization': '新结构初始化',
          'N0_runtime_audit':'端到端推理接入检查','P0_full_capture':'补齐29层原始数据',
          'N1_seal':'封存视频对照权重','N2_videos':'闭环生成与像素质量评估',
          'N3_report':'整理视频和端到端报告','all_finished':'端到端测试完成',
          'B15_calibration': '训练集校准15%保留率', 'B15_lock': '锁定稀疏预算',
          'P3_preflight': '真实注意力损失与梯度检查',
          'K_retraining': '基础路由训练', 'P0_real': '真实数据检查',
          'P1_baseline': '基础质量与成本扫描', 'P2_projection': 'Value 投影拟合',
          'P2_features': '动作标签生成', 'P3_training': '四组配对训练',
          'P3_report': '完整验证与成本曲线', 'P3_merge': '整理评估报告'}
WORKER_STATES = {'training': '训练中', 'paired_extension_wait': '等待配对决策',
                 'completed': '已完成', 'evaluating': '验证中', 'failed': '失败',
                 'loading': '载入中', 'initializing': '初始化', 'paused': '已暂停',
                 'fitting': '拟合中', 'generating': '生成中'}
WORKER_STATES['collecting']='采集中'


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def stamp(value):
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, AttributeError):
        return None


def width(value):
    plain = re.sub(r'\x1b\[[0-9;]*m', '', str(value))
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in plain)


def table(headers, rows):
    all_rows = [headers, *rows]
    sizes = [max(width(row[i]) for row in all_rows) for i in range(len(headers))]
    return ['  '.join(str(v) + ' ' * (n-width(v)) for v, n in zip(row, sizes)).rstrip()
            for row in all_rows]


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '估算中'
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}秒'
    minutes = math.ceil(seconds/60)
    hours, minutes = divmod(minutes, 60)
    return (f'{hours}小时' if hours else '') + (f'{minutes}分' if minutes else '')


def progress(done, total, size=22):
    ratio = min(1., max(0., done/max(total, 1)))
    count = int(ratio*size)
    return '█'*count + '░'*(size-count) + f'  {done}/{total}  {ratio:.0%}'


def process_matches(pid, directory):
    if not pid:
        return None
    try:
        process = Path('/proc')/str(pid)
        if (process/'stat').read_text().split(') ', 1)[1].startswith('Z'):
            return False
        return str(directory) in (process/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, ValueError):
        return False


def snapshot(root, now=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    m = read(root/'experiment.json')
    state = read(root/'status.json')
    groups = m.get('groups', list(GROUP_NAMES))
    layers = m.get('layers', [])
    seeds = m.get('training_seeds', [0, 1])
    expected = {(layer, group, seed) for layer in layers for group in groups for seed in seeds}
    completed = {}
    for path in (root/'P3_training').glob('layer*/*/seed*/completed.json'):
        value = read(path)
        key = (value.get('layer'), value.get('kind'), value.get('seed'))
        if key in expected:
            completed[key] = value
    worker_count=m.get('worker_count',len(groups)*len(seeds))
    workers = [read(root/f'P3_training/rank{i}.json') for i in range(worker_count)]
    train_launch = read(root/'P3_training_workers/launcher_status.json')
    launcher_ranks = {x['rank']: x for x in train_launch.get('ranks', [])}
    gpu_names = train_launch.get('gpus', list(range(len(workers))))
    active = {}
    for rank, worker in enumerate(workers):
        if not worker:
            continue
        worker['gpu'] = gpu_names[rank] if rank < len(gpu_names) else rank
        worker['alive'] = process_matches(launcher_ranks.get(rank, {}).get('pid'), root)
        key = (worker.get('layer'), worker.get('kind'), worker.get('seed'))
        if key in expected and key not in completed and worker.get('updates') is not None:
            active[key] = worker
    done_steps = sum(x.get('updates', 0) for x in completed.values()) + sum(x['updates'] for x in active.values())
    low_left = high_left = 0
    for key in expected-completed.keys():
        value = active.get(key, {})
        step = value.get('updates', 0)
        low_left += max(0, value.get('cap', m.get('min_updates', 400))-step)
        high_left += max(0, m.get('max_updates', 800)-step)
    started = stamp(train_launch.get('started_at'))
    train_elapsed = max(0, now-started) if started else None
    rate = done_steps/train_elapsed if train_elapsed and done_steps else None
    eta = (low_left/rate, high_left/rate) if rate and train_elapsed >= 20 else None
    feature_states = [read(root/f'P2_features/layer{layer:02d}.json') for layer in layers]
    per_layer = sum(r.get('history_tokens', 0) > 0 for r in m.get('rows', []))
    feature_done = sum(per_layer if d.get('status') == 'completed' else d.get('completed', 0)
                       for d in feature_states)
    report_workers = [read(root/f'P3_reports/rank{i}.json') for i in range(len(workers))]
    report_total = len(expected)*2*len(m.get('beta_grid', []))*len(m.get('lambda_grid', []))*len(m.get('report_margins',['zero','train_q90']))
    report_done = sum(d.get('points', d.get('completed_points', 0)) for d in report_workers)
    phase_workers=[]
    phase_alias={'P0_full_capture':'capture','K_initialization':'fit','K_retraining':'foundation'}
    phase=phase_alias.get(state.get('stage'),state.get('stage'))
    for rank in range(len(workers)):
        value=read(root/f'{phase}/rank{rank}.json')
        if not value:
            values=[dict(read(root/f'{phase}/layer{layer:02d}.json'),layer=layer)for layer in layers[rank::worker_count]]
            values=[v for v in values if 'status' in v]
            value=next((v for v in values if v.get('status')!='completed'),values[-1] if values else {})
        phase_workers.append(value)
    dates = [stamp(d.get('updated_at')) for d in [*workers, *feature_states, *report_workers, *phase_workers, state]]
    dates = [d for d in dates if d is not None]
    video_rows=[read(p)for p in (root/'videos').glob('*/result.json')]
    sealed=read(root/'e2e_manifest.json')
    video_total=len(sealed.get('prompt_ids',[]))*len(sealed.get('generation_seeds',[]))
    return dict(root=root, config=m, state=state, workers=workers, completed=completed,
        expected=expected, active=active, done_steps=done_steps, train_elapsed=train_elapsed,
        eta=eta, feature_done=feature_done, feature_total=per_layer*len(layers),
        report_done=report_done, report_total=report_total, report_workers=report_workers,
        phase_workers=phase_workers,
        video_done=len(video_rows),video_total=video_total,
        raw_done=len(list((Path(m['cache'])/'receipts').glob('*.json'))) if m.get('cache') else 0,
        foundation_done=len(list((root/'parameters').glob('layer*.pt'))),
        full_layers=[layer for layer in layers if all((layer,g,s) in completed for g in groups for s in seeds)],
        controller_alive=process_matches(state.get('controller_pid'), root),
        age=max(0, now-max(dates)) if dates else None, now=now)


def validation(data):
    """Compare one common layer/step, never mix differently scaled layers."""
    root, m = data['root'], data['config']
    if data['full_layers']:
        layer = data['full_layers'][-1]
        records = {}
        filename='best_validation.json' if m.get('best_metric')=='nmse' else 'final_validation.json'
        for group in m['groups']:
            for seed in m['training_seeds']:
                d = read(root/f'P3_training/layer{layer:02d}/{group}/seed{seed}'/filename)
                if not d:
                    return None
                records[group, seed] = d
        suffix=' · best' if m.get('best_metric')=='nmse' else ''
        label = f'最近完成层 {layer} · 全量96事件 · 两个种子均值'+suffix
    else:
        current = [d for d in data['workers'] if d.get('layer') is not None]
        if not current or len({d['layer'] for d in current}) != 1:
            return None
        layer = current[0]['layer']
        curves = {(g,s): read(root/f'P3_training/layer{layer:02d}/{g}/seed{s}/validation.json').get('curve', [])
                  for g in m['groups'] for s in m['training_seeds']}
        steps = [set(x['updates'] for x in c) for c in curves.values()]
        common = set.intersection(*steps) if steps else set()
        if not common:
            return None
        step = max(common)
        records = {key: next(x for x in curve if x['updates'] == step) for key,curve in curves.items()}
        events=96 if m.get('monitor_events')=='all96' else 24
        label = f'当前层 {layer} · 共同验证点{step}步 · 监测{events}事件'
    rows = []
    for group in m['groups']:
        values = [records[group, seed] for seed in m['training_seeds']]
        rows.append([GROUP_NAMES.get(group, group),
            f'{sum(v["nmse"] for v in values)/len(values):.6f}',
            f'{sum(v["keep"] for v in values)/len(values):.2%}'])
    return label, rows


def render(data, details=False, color=False, columns=100, rows=40, watching=False, interval=5):
    def paint(text, code):
        return f'\033[{code}m{text}\033[0m' if color else text

    compact = watching and rows < 32
    m, state = data['config'], data['state']
    if not m:
        return '实验配置暂不可读，请稍后刷新。'
    stage, status = state.get('stage'), state.get('status')
    e2e=m.get('end_to_end',{}).get('enabled',False)
    stage_name = STAGES.get(stage, stage or '等待启动')
    if stage=='P3_training':stage_name=f'{len(m["groups"])}组×{len(m["training_seeds"])}种子配对训练'
    title = datetime.fromtimestamp(data['now'], timezone.utc).strftime('%H:%M:%S UTC')
    lines = [paint('Slim-AV v0.10  实验看板', '1;36') + '    ' + title]
    lines.append('─' * min(88, max(52, columns)))
    if status=='completed':
        lines.append(paint('✓ 端到端测试、视频和报告全部完成','1;32'))
    elif status == 'failed':
        lines.append(paint('● 发生错误 · ' + stage_name, '1;31'))
    elif status == 'paused':
        lines.append(paint('● 已暂停 · ' + stage_name, '1;33'))
    elif status == 'phase_complete':
        lines.append(paint('✓ 本轮离线队列已完成' if state.get('through') == 'P3'
                           else '✓ 基础扫描已完成，等待下一阶段', '1;32'))
    elif status == 'running' and data['controller_alive'] is False:
        lines.append(paint('! 控制进程已离线 · 磁盘记录停留在 ' + stage_name, '1;31'))
    else:
        lines.append(paint('● 正在进行：' + stage_name, '1;32'))
    fresh = '尚无更新时间' if data['age'] is None else '刚更新' if data['age'] < 3 else duration(data['age'])+'前更新'
    scope=f'{len(m["layers"])}层端到端流水线' if e2e else f'{len(m["layers"])}层离线实验，尚非端到端视频结果'
    lines.append('数据' + fresh + '  ·  当前范围：'+scope)
    lines.append('原型数：历史近/中/远 8/4/2 · 当前16；首chunk Dense，后续不强制保留')
    if m.get('training_objective')=='actual_av_nmse':
        beta=m.get('monitor_beta') if m.get('budget_locked') else '校准中'
        lines.append(f'目标保留15%（后续chunk） · β={beta} · Loss/best：真实注意力误差')
    total = len(data['expected'])
    training_done = len(data['completed'])
    flags = [data['raw_done'] == len(m.get('rows', [])),
             read(data['root']/'P0_math.json').get('passed', False),
             data['foundation_done'] == len(m['layers']),
             data['feature_done'] == data['feature_total'],
             training_done == total,
             bool(data['report_total']) and data['report_done'] >= data['report_total'],
             (data['root']/'reports/summary.json').exists()]
    names = ['数据', '审计', '基础训练', '标签', '配对训练', '评估', '报告']
    active_index = 0 if stage in ('P0_raw_pilot', 'P0_full_capture') else 1 if stage in ('P0_math', 'P0_real','B15_calibration','B15_lock') else 2 if stage in ('K_initialization','K_retraining') else 3 if stage in ('P1_baseline','P2_projection','P2_features','P3_preflight') else 4 if stage == 'P3_training' else 5 if stage == 'P3_report' else 6
    if e2e:
        names=['数据','审计','基础训练','标签','配对训练','视频','报告']
        flags[-2]=bool(data['video_total'])and data['video_done']==data['video_total']
        flags[-1]=(data['root']/'video_review/summary.json').exists()
        if stage in ('N0_runtime_audit',):active_index=1
        elif stage in ('N1_seal','N2_videos'):active_index=5
        elif stage=='N3_report':active_index=6
    chips = [('✓' if done else '▶' if i == active_index and status == 'running' else '○') + name
             for i,(name,done) in enumerate(zip(names,flags))]
    lines += ['', ' → '.join(chips)]
    lines.append('动作头完成  ' + paint(progress(training_done, total), '36'))
    layer_text = ', '.join(map(str, data['full_layers'])) or '暂无'
    if watching and len(data['full_layers'])>8:
        layer_text=', '.join(map(str,data['full_layers'][-3:]))+f'等{len(data["full_layers"])}层'
    lines.append(f'整层完成    {len(data["full_layers"])}/{len(m["layers"])} 层；已完成层：{layer_text}')
    if e2e and stage in ('N1_seal','N2_videos','N3_report','all_finished'):
        lines.append('端到端视频  '+progress(data['video_done'],data['video_total']))
    if stage == 'P2_features':
        lines.append('动作标签    ' + progress(data['feature_done'], data['feature_total']))
    elif stage=='K_retraining':
        lines.append('K基础参数   '+progress(data['foundation_done'],len(m['layers'])))
    elif stage == 'P3_report':
        lines.append('完整评估    ' + progress(data['report_done'], data['report_total']))
    if stage == 'P3_training':
        eta = data['eta']
        estimate = '首批更新后估算' if eta is None else duration(eta[0])+'～'+duration(eta[1])
        if status != 'running' or data['controller_alive'] is False:
            estimate = '暂停估算'
        lines.append(f'训练已用 {duration(data["train_elapsed"])}  ·  剩余约 {estimate}')
        if not compact:
            lines.append('时间按400～800步预算估计；后续完整评估耗时另计。')
    worker_rows = []
    generic = stage not in ('P3_training','P3_report','P3','P3_merge')
    for rank, worker in enumerate(data['workers']):
        if generic:
            worker=data['phase_workers'][rank]
            if not worker:
                continue
            label=WORKER_STATES.get(worker.get('status'),'运行中')
            layer=worker.get('layer','全层' if stage in ('P0_full_capture','N2_videos') else '—')
            total_events=worker.get('total',data['feature_total']//max(len(m['layers']),1) if stage=='P2_features' else '—')
            count=f'{worker.get("completed", "—")}/{total_events}' if worker.get('status')!='completed' else '已完成'
            if 'updates' in worker and worker.get('status')!='completed':
                count=f'{worker["updates"]}/{m.get("foundation_updates",400)}步'
            if stage=='N2_videos':
                label+=(' · '+str(worker['mode'])) if worker.get('mode') else ''
            worker_rows.append([rank,layer,count,label])
            continue
        if stage == 'P3_report':
            worker = dict(data['report_workers'][rank])
            worker['gpu'] = rank
        kind = worker.get('kind', m['groups'][rank % len(m['groups'])])
        seed = worker.get('seed', (rank // len(m['groups']))%len(m['training_seeds']))
        label = WORKER_STATES.get(worker.get('status'), '等待启动')
        if worker.get('status') in ('training','paired_extension_wait') and worker.get('alive') is False:
            label = '进程已离线'
        if stage == 'P3_report':
            step = str(worker.get('points', worker.get('completed_points', 0)))+'点'
        else:
            step = f'{worker["updates"]}/{worker.get("cap",400)}' if 'updates' in worker else '—'
        worker_rows.append([worker.get('gpu',rank), GROUP_NAMES.get(kind,kind), seed,
                            worker.get('layer','—'), step, label])
    if not compact:
        lines += ['', paint('并行任务', '1')]
    if generic:
        if worker_rows:
            lines += table(['卡','层','进度','状态'],worker_rows)
        else:
            lines += ['正在准备本阶段任务…']
    else:
        lines += table(['卡', '对照组', '种子', '层', '更新/上限' if stage != 'P3_report' else '评估点', '状态'], worker_rows)
    extra = []
    v = validation(data)
    if v:
        extra += ['', paint(v[0], '1'), *table(['对照组','注意力误差 NMSE ↓','KV保留率'], v[1]),
                  '局部开发验证；不同保留率需结合成本曲线比较。']
    if details:
        cells = []
        for layer in m['layers']:
            values = []
            for group in m['groups']:
                pair = []
                for seed in m['training_seeds']:
                    key = (layer,group,seed)
                    if key in data['completed']:
                        pair.append(str(data['completed'][key].get('updates',''))+'✓')
                    elif key in data['active']:
                        pair.append(str(data['active'][key]['updates'])+'▶')
                    else:
                        pair.append('·')
                values.append('/'.join(pair))
            cells.append([layer,*values])
        extra += ['', paint('逐层进度：每格为种子0/种子1；✓完成 ▶进行 ·待运行', '1')]
        extra += table(['层',*[GROUP_NAMES.get(g,g) for g in m['groups']]],cells)
        extra += ['', f'数据补采：{data["raw_done"]}/{len(m.get("rows",[]))}  ·  基础训练：{data["foundation_done"]}/{len(m["layers"])}层',
                  '实验目录：'+str(data['root']), '阶段日志：'+str(data['root']/(str(stage)+'_workers'))]
    error = state.get('error')
    if error:
        lines += ['', paint(str(error), '31')]
    if watching and len(lines)+len(extra)+2 > rows:
        if rows > len(lines)+3:
            lines.append('放大窗口可显示验证指标；--once --details 可查看完整详情。')
    else:
        lines += extra
    lines += ['', f'每{interval:g}秒原地刷新 · Ctrl+C 只退出查看' if watching
              else '实时查看：./v10_progress.sh   详细快照：./v10_progress.sh --once --details']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='v0.10 中文实验看板；终端默认实时刷新')
    parser.add_argument('--watch', action='store_true', help='实时原地刷新')
    parser.add_argument('--once', action='store_true', help='仅打印一次')
    parser.add_argument('--details', action='store_true', help='显示逐层任务和日志路径')
    parser.add_argument('--no-color', action='store_true', help='关闭颜色')
    parser.add_argument('--interval', type=float, default=5., help='刷新间隔（秒），默认5')
    parser.add_argument('--directory', type=Path, help='指定实验目录')
    args = parser.parse_args()
    if args.interval < .5:
        parser.error('刷新间隔至少0.5秒')
    if args.once and args.watch:
        parser.error('--once 与 --watch 不能同时使用')
    root = args.directory or Path(read(ROOT/'migration/unified_router_active.json').get('directory', ''))
    if not (root/'experiment.json').is_file():
        parser.exit(1, '未找到实验配置；可用 --directory 指定目录。\n')
    interactive = sys.stdout.isatty() and os.environ.get('TERM') != 'dumb'
    watching = args.watch or (interactive and not args.once)
    alternate = watching and interactive
    color = interactive and not args.no_color and 'NO_COLOR' not in os.environ
    try:
        if alternate:
            sys.stdout.write('\033[?1049h\033[?25l')
        while True:
            columns, rows = shutil.get_terminal_size((100, 40))
            output = render(snapshot(root), args.details, color, columns, rows, watching, args.interval)
            sys.stdout.write(('\033[H' if alternate else '') + output + ('\033[J' if alternate else '') + '\n')
            sys.stdout.flush()
            if not watching:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if alternate:
            sys.stdout.write('\033[?25h\033[?1049l')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
