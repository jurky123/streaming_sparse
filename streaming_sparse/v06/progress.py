"""Readable Chinese progress, no CUDA/PyTorch imports."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render():
    path = HERE / 'reports/workflow/status.json'
    if not path.exists():
        return 'v0.6：工作流尚未启动。\n'
    state = json.loads(path.read_text()); jobs = state['jobs']; counts = Counter(v['status'] for v in jobs.values())
    lines = ['v0.6 Relevance / Interaction 分离验证',
             '更新时间：' + state['updated_utc'],
             f"任务：完成 {counts['completed']}，运行 {counts['running']}，等待 {counts['pending']}，条件跳过 {counts['skipped']}，失败 {counts['failed']}",
             'GPU：每张卡最多一个任务；不使用代理。', '']
    labels = {'stage1': 'Stage I：独立训练 Relevance-only C0', 'stage2': 'Stage II：冻结同一 C0，独立训练 C1/C2/C3',
              'stage3': 'Stage III：仅验证集收益稳定的结构联合微调', 'scratch': 'Scratch：与 Staged 匹配总更新预算的从头联合训练',
              'signed': 'Signed 诊断：实际重训练关系消融'}
    for name, job in jobs.items():
        if job['status'] not in ('running', 'failed'):
            continue
        lines.append(f"GPU {job.get('gpu')} | {name} | {job['status']} | PID {job.get('pid')}")
        run = job.get('run_directory')
        if run:
            status = Path(run) / 'status.json'
            if status.exists():
                x = json.loads(status.read_text()); maximum = x['max_updates']; step = x['updates']
                lines.append(f"  {labels.get(x['stage'], x['stage'])}；seed={x['seed']}；{x['status']}")
                lines.append(f'  更新：{step:,}/{maximum:,}（上限进度 {100*step/max(maximum,1):.1f}%；可能按验证集平台提前停止）')
                for arm, score in x['best_validation'].items():
                    if score is not None:
                        lines.append(f"  {arm} 最佳 Val Global Error={score:.7f}，step={x['best_step'][arm]:,}")
            for split in ('val', 'test'):
                target = Path(run) / f'full_{split}/progress.json'
                if target.exists() and ('eval' in name or 'full' in name):
                    x = json.loads(target.read_text()); lines.append(f"  {split}：{x['completed']}/{x['total']} snapshots")
        lines.append('  日志：' + str(HERE / f'reports/workflow/{name}.log'))
    decision = HERE / 'reports/stage2_decision.json'
    if decision.exists():
        x = json.loads(decision.read_text()); lines += ['', 'Validation 锁定的 Stage III 结构：' + (', '.join(x['eligible']) or '无'),
                    'Signed 关系重训练：' + ('触发' if x['signed_retraining_required'] else '无需触发')]
    else:
        lines += ['', '后续队列：3 个 seed 的 C0 → C1/C2/C3 → 全量 Val → 条件 Stage III / Scratch / Signed 消融 → 开发 Test 与离线诊断。']
    if state.get('aborted'):
        lines.append('工作流已因任务失败停止发出新任务；请查看失败日志。')
    lines += ['', '注意：旧 Test 已参与前期研究设计，只作开发测试；不是新的独立发表级 Test。']
    return '\n'.join(lines) + '\n'


def main():
    print(render(), end='')


if __name__ == '__main__':
    main()
