"""Readable progress without importing Torch or initializing CUDA."""
import json
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent


def render():
    path=HERE/'reports/workflow/status.json'
    if not path.exists():return 'CAVIR v0.7：尚未启动\n'
    s=json.loads(path.read_text());counts=Counter(j['status'] for j in s['jobs'].values())
    lines=['CAVIR v0.7 离线实验',f"更新：{s['updated_utc']}",
           f"完成 {counts['completed']} | 运行 {counts['running']} | 等待 {counts['pending']} | 条件跳过 {counts['skipped']} | 失败 {counts['failed']}",
           'v0.6 后续派发已暂停；正在运行的旧任务完成后，空闲 GPU 优先给 v0.7。',
           'GPU 状态：'+json.dumps(s.get('gpu_occupancy',{}),ensure_ascii=False),'']
    for name,j in s['jobs'].items():
        if j['status'] not in ('running','failed'):continue
        lines.append(f"{name} | GPU {j.get('gpu')} | {j['status']} | PID {j.get('pid')}")
        directory=j.get('directory')
        if directory:
            run=Path(directory)
            status=run/'status.json'
            if status.exists():
                v=json.loads(status.read_text());lines.append(f"  {v['phase']} seed={v['seed']} {v['variant'] or ''} {v['status']} {v['updates']:,}/{v['max_updates']:,}")
                lines.append('  最佳验证：'+json.dumps(v['best'],ensure_ascii=False))
            for p in (run/'progress.json',):
                if p.exists():
                    v=json.loads(p.read_text());lines.append(f"  评估 snapshots：{v['completed']}/{v['total']}")
        lines.append('  日志：'+str(HERE/f'reports/workflow/{name}.log'))
    for name in ('teacher_decision','sketch_decision'):
        p=HERE/f'reports/{name}.json'
        if p.exists():
            d=json.loads(p.read_text());lines.append(f"{name}："+json.dumps({k:d[k] for k in ('headroom','beta','sufficient') if k in d},ensure_ascii=False))
    if s['aborted']:lines.append('任务失败，停止发出新任务。')
    lines += ['', '实验：Phase0 → T1/T2/T3 真值诊断 → S1/S4/S8 → R1/R2/R3/等容量Local/R4条件对照 → 三预算全量Val。',
              '当前复用现有 Train/Val。旧 Test 已曝光，不作为新的独立最终 Test；本轮不做 Kernel/加速/视频质量/Backbone训练。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':print(render(),end='')
