"""Read-only experiment dashboard; process liveness takes priority over stale JSON."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[3] / 'streaming_sparse/v10/experiments/opt20_all29_tile_finetune_20261025'
NAMES = {'training': '训练', 'validating': '验证', 'loading': '加载数据',
         'initial_validation': '初始验证', 'completed': '完成', 'failed': '失败',
         'interrupted': '进程退出', 'pending': '等待调度', 'stopped': '已停止'}
ACTIVE = {'training', 'validating', 'loading', 'initial_validation'}


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def stamp(d):
    try:
        return datetime.fromisoformat(d.get('updated_at', '')).timestamp()
    except ValueError:
        return 0


def alive(pid):
    try:
        fields = (Path('/proc') / str(int(pid)) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] not in ('Z', 'X')
    except (OSError, ValueError, TypeError):
        return False


def classify(d, completed, worker, launch, code=None):
    if completed:
        return 'completed'
    if worker and worker.get('status') == 'running':
        if not worker.get('alive'):
            return 'failed' if d.get('status') == 'failed' and stamp(d) >= worker.get('started_at', launch) else 'interrupted'
        if stamp(d) < worker.get('started_at', launch):
            return 'loading'
        return d.get('status') if d.get('status') in ACTIVE | {'failed', 'stopped'} else 'loading'
    if stamp(d) < launch:
        return 'pending'
    if d.get('status') == 'failed' or code not in (None, 0):
        return 'failed'
    if d.get('status') in ACTIVE:
        return 'interrupted'
    return 'stopped' if d.get('status') == 'stopped' else 'pending'


def snapshot():
    launch = read(ROOT / 'training/launcher.json')
    started = launch.get('started_at', 0)
    workers = []
    for rank in range(8):
        w = read(ROOT / f'training/worker{rank}.json')
        w.update(rank=rank, alive=alive(w.get('pid')))
        if w.get('started_at', started) < started:
            w = dict(rank=rank, alive=False, status='pending')
        workers.append(w)
    mapping = {(w.get('layer'), w.get('variant')): w for w in workers if w.get('status') == 'running'}
    codes_path = ROOT / 'training/exit_codes.json'
    codes = read(codes_path) if codes_path.exists() and codes_path.stat().st_mtime >= started else {}
    jobs = []
    for layer in range(1, 30):
        for variant in ['Tile15', 'Tile20']:
            folder = ROOT / f'training/layer{layer:02d}/{variant}'
            raw = read(folder / 'status.json')
            done = read(folder / 'completed.json')
            w = mapping.get((layer, variant))
            state = classify(raw, done, w, started, codes.get(f'layer{layer:02d}_{variant}'))
            data = dict(raw) if state != 'pending' else {}
            data.update(done)
            data.update(layer=layer, variant=variant, state=state, gpu=w['rank'] if w else None,
                        stale_record=bool(raw and stamp(raw) < started and not done))
            jobs.append(data)
    workflow = read(ROOT / 'workflow.json')
    return dict(workflow=workflow, workflow_alive=alive(workflow.get('pid')),
                launcher_alive=alive(launch.get('pid')), workers=workers, jobs=jobs,
                counts=dict(Counter(j['state'] for j in jobs)),
                before=len(list((ROOT / 'e2e_before').glob('*.seed0/completed.json'))),
                final=len(list((ROOT / 'e2e_final').glob('*.seed0/completed.json'))))


def pad(value, width):
    text = str(value)
    size = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)
    return text + ' ' * max(0, width - size)


def table(headers, rows, widths):
    print('  '.join(pad(v, w) for v, w in zip(headers, widths)))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print('  '.join(pad(v, w) for v, w in zip(row, widths)))


def number(value):
    return f'{value:.6f}' if value is not None else '—'


def gain(d):
    return f"{100*(1-d['best']/d['initial']):+.1f}%" if d.get('initial') and d.get('best') is not None else '—'


def compact(jobs, state):
    result = []
    for variant in ['Tile15', 'Tile20']:
        layers = sorted(j['layer'] for j in jobs if j['variant'] == variant and j['state'] == state)
        groups = []
        for layer in layers:
            if groups and layer == groups[-1][-1] + 1:
                groups[-1].append(layer)
            else:
                groups.append([layer])
        if groups:
            result.append(variant + ' L' + ','.join(str(g[0]) if len(g) == 1 else f'{g[0]}–{g[-1]}' for g in groups))
    return ' / '.join(result) or '无'


def gpu_stats():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
        return {int(x[0]): f'{x[1]}% / {int(x[2])/1024:.1f}G' for line in out.stdout.splitlines() if len(x := [v.strip() for v in line.split(',')]) == 4}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def show(all_layers=False, no_gpu=False):
    s = snapshot(); jobs = s['jobs']; counts = s['counts']
    done = counts.get('completed', 0); running = sum(counts.get(k, 0) for k in ACTIVE)
    failed = counts.get('failed', 0); interrupted = counts.get('interrupted', 0)
    stage = s['workflow'].get('status', 'unknown')
    phases = {'training_and_before_video': '全层联合微调', 'final_video': '微调后端到端评估',
              'completed': '全部完成', 'training_failed': '训练有失败', 'final_video_failed': '视频评估失败', 'report_failed': '报告生成失败'}
    print('Opt20 全层微调  |  实际 tile 预算 15% / 20%  |  ' + datetime.now(timezone.utc).strftime('%H:%M:%S UTC'))
    print('层1–29：prototype + key投影联合训练；层0/首chunk Dense；主干冻结。')
    state = '存活' if s['workflow_alive'] else ('已结束' if stage == 'completed' else '已退出')
    print(f"阶段：{phases.get(stage, stage)}  |  主控：{state}  |  训练调度：{'存活' if s['launcher_alive'] else '未运行'}")
    print(f"完成 [{('#' * (done*24//58)).ljust(24, '.')}] {done}/58  |  运行 {running}  等待 {counts.get('pending',0)}  失败 {failed}  中断 {interrupted}  停止 {counts.get('stopped',0)}")
    video_note = '全部完成' if s['final'] == 8 else '训练完成后自动开始'
    print(f"视频：微调前 {s['before']}/8  →  微调后 {s['final']}/8（{video_note}）")
    if not s['workflow_alive'] and stage != 'completed':
        print('! 主控进程未存活：旧状态文件不代表任务仍在运行。')
    stats = {} if no_gpu else gpu_stats()
    rows = []
    for w in s['workers']:
        rank = w['rank']; j = next((j for j in jobs if j['gpu'] == rank), {})
        if j:
            age = max(0, int(time.time() - stamp(j))) if stamp(j) else None
            rows.append([rank, f"L{j['layer']:02d} {j['variant']}", NAMES.get(j['state'], j['state']),
                         f"{j.get('updates',0)}/{j.get('total',2000)}", number(j.get('losses',{}).get('actual_nmse')),
                         number(j.get('best')), gain(j), f'{age}s' if age is not None else '—', stats.get(rank, '—')])
        else:
            label = '视频生成' if w.get('status') == 'video_before' and w['alive'] else '空闲'
            if stage == 'final_video':
                video = read(ROOT / f'e2e_final/rank{rank}.json')
                label = {'quality':'视频质量','timing':'计时','warmup':'预热','loading':'加载模型','completed':'视频完成'}.get(video.get('status'), label)
                if not s['workflow_alive'] and video.get('status') != 'completed': label = '待检查'
            rows.append([rank, '—', label, '—', '—', '—', '—', '—', stats.get(rank, '—')])
    print()
    table(['GPU', '当前任务', '状态', '步数', 'Train NMSE', 'Val best', '下降', '更新前', '利用率/显存'], rows, [3, 12, 8, 10, 11, 11, 7, 7, 13])
    print('\n等待调度：' + compact(jobs, 'pending'))
    print('已完成：' + compact(jobs, 'completed'))
    stale = sum(j['stale_record'] for j in jobs if j['state'] == 'pending')
    if stale:
        print(f'说明：{stale} 组含上一轮旧记录，本轮尚未调度；统一显示“等待调度”，不计入本轮失败。')
    for j in jobs:
        if j['state'] in {'failed', 'interrupted'}:
            print(f"! L{j['layer']:02d} {j['variant']} {NAMES[j['state']]}：{j.get('error','工作进程已退出，可从 recovery.pt 恢复')}")
            print(f"  日志：{ROOT}/training/layer{j['layer']:02d}_{j['variant']}.log")
    if all_layers:
        print('\n全部层：状态 / 步数 / Val best / 相对初始误差下降')
        rows = []
        for layer in range(1,30):
            cells = []
            for v in ['Tile15','Tile20']:
                j = next(j for j in jobs if j['layer']==layer and j['variant']==v)
                cells.append('等待调度' if j['state']=='pending' else f"{NAMES.get(j['state'],j['state'])} {j.get('updates',0):4}/2000  {number(j.get('best'))}  {gain(j)}")
            rows.append([f'L{layer:02d}', *cells])
        table(['层','Tile15','Tile20'], rows, [3, 39, 39])
    if stage == 'completed':
        result = read(ROOT / 'summary.json')
        quality = result.get('quality', {})
        dense = result.get('dense_generation_seconds')
        if dense and quality:
            print(f'\n最终视频结果：8条配对验证；LPIPS统计后36帧；Dense生成 {dense:.4f}s（不含VAE）')
            rows = []
            for variant in ['Tile15', 'Tile20']:
                before = quality[variant + '_before']; after = quality[variant + '_after']
                rows.append([variant, f"{100*after['executed']:.2f}%", f"{before['lpips']:.4f} → {after['lpips']:.4f}",
                             f"{after['generation_seconds']:.4f}s", f"{dense/after['generation_seconds']:.3f}x"])
            table(['方案','attention执行率','LPIPS 微调前 → 后 ↓','生成耗时','相对Dense加速'], rows, [8,17,23,10,14])
            print('视频对比：' + str(ROOT / 'index.html'))
    print('\nTrain 为当前批次；Val best 为实际 attention 输出 NMSE；下降相对各组初始值。')
    print('用法：bash opt20_status.sh --watch  |  全层详情：加 --all  |  Ctrl+C 退出查看')
    print('实验目录：' + str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch', action='store_true', help='自动刷新')
    p.add_argument('--all', action='store_true', help='显示全部29层')
    p.add_argument('--interval', type=float, default=5, help='刷新间隔秒数，默认5')
    p.add_argument('--no-gpu', action='store_true', help='不查询GPU利用率')
    p.add_argument('--json', action='store_true', help='输出机器可读快照')
    a = p.parse_args()
    if a.interval < 1: p.error('--interval 必须至少为1秒')
    try:
        while True:
            if a.watch and sys.stdout.isatty(): print('\033[2J\033[H', end='')
            if a.json: print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
            else: show(a.all, a.no_gpu)
            if not a.watch: break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        sys.stdout = open('/dev/null', 'w')


if __name__ == '__main__':
    main()
