"""Human-readable progress; no torch import required."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / 'streaming_sparse/v10/experiments/opt12_joint_20261025'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    root = ROOT
    m = read(root / 'experiment.json')
    print('Opt12 联合微调 | prototype 分配 + K64 投影 + 动作头 | 8×A100')
    print('实际优化内核注意力 NMSE；首 chunk、layer0 Dense；保留率约15%；主模型冻结。')
    print('两组：Compact / Aggressive(Skip0+FP16 PV)。每层最多2000步，至少1000步后才按平台期停止。\n')
    assignments = {x['rank']: x['layers'] for x in read(root / 'training/launcher.json').get('workers', [])}
    completed = list((root / 'training').glob('layer*/completed.json'))
    phases = {'loading': '加载数据', 'initial_validation': '初始验证', 'training': '联合训练',
              'validating': '完整验证', 'completed': '已完成', 'failed': '失败', 'stopped': '已停止'}
    remaining = []
    print(' GPU  当前层   更新进度       状态          最近一步耗时')
    for rank in range(8):
        d = read(root / f'training/rank{rank}.json')
        step = d.get('updates', 0)
        layer = d.get('layer', '-')
        seconds = d.get('update_seconds')
        timing = f'{seconds:.2f}s' if seconds else '-'
        print(f'  {rank}    {str(layer):>3}    {step:4}/{m.get("max_updates", 2000)}    {phases.get(d.get("status"), "等待"):8}    {timing}')
        if d.get('status') == 'failed':
            print('      ' + d.get('error', ''))
        for variant, metrics in d.get('losses', {}).items():
            best = d.get('best', {}).get(variant)
            initial = d.get('initial', {}).get(variant)
            improve = f'{(1-best/initial)*100:+.3f}%' if best is not None and initial else '-'
            print(f'      {variant:10} train NMSE={metrics["actual_nmse"]:.6f}  保留={metrics["keep"]:.2%}  Val best改善={improve}')
        if seconds and isinstance(layer, int):
            assigned = assignments.get(rank, m['layers'][rank::8])
            future = sum(not (root / f'training/layer{i:02d}/completed.json').exists() for i in assigned if i != layer)
            remaining.append(seconds * ((m['max_updates'] - step) + future * m['max_updates']))
    print(f'\n已完成层：{len(completed)}/29（每层含两组）；预检：{"通过" if read(root / "preflight.json").get("passed") else "未通过"}')
    if remaining:
        print(f'按当前步速估计剩余训练计算：约{max(remaining)/3600:.1f}小时，另需验证/加载时间；早停会缩短。')
    for v in m.get('variants', []):
        before, after = [], []
        for i in m['layers']:
            a = read(root / f'training/layer{i:02d}/{v}/initial_validation.json')
            b = read(root / f'training/layer{i:02d}/{v}/best_validation.json')
            if a and b:
                before.append(a['nmse'])
                after.append(b['nmse'])
        if before:
            print(f'{v}: 已验证{len(before)}层，平均实际 NMSE {sum(before)/len(before):.6f} → {sum(after)/len(after):.6f}')
    codes = read(root / 'training/exit_codes.json')
    if codes:
        print('进程退出码：', codes)
    print('\n实验目录：' + str(root))
    print('完整训练曲线：training/layerXX/{Compact,Aggressive}/train.jsonl')
    print('原始验证曲线：training/layerXX/{Compact,Aggressive}/curve.json')


if __name__ == '__main__':
    main()
