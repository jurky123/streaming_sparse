"""Read-only paired-router progress; no Torch import or GPU context."""
import argparse
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    try:return json.loads(Path(path).read_text())
    except(FileNotFoundError,ValueError):return{}


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    root=Path(read(ROOT/'migration/paired_router_active.json')['directory'])
    while True:
        config=read(root/'experiment.json');state=read(root/'status.json')
        if a.watch:print('\033[H\033[2J',end='')
        print(f"配对 Router 主训练 | {state.get('status')} | {state.get('stage')} | 骨干/LoRA冻结")
        print('同一起点、同一数据次序；每组batch8、4卡；每组2000–4000更新，配对验证决定停止。')
        for stage in ['three','dense0']:
            spec=config['stages'][stage];directory=Path(spec['directory']);print(f"\n{stage}: {len(spec['layers'])}个稀疏层 / {12*len(spec['layers'])}个独立Layer×Head Router")
            for group in ['F-K','F-V']:
                run=directory/f'runs/seed0/{group}';checkpoint=read(run/'recovery.json');curve=read(run/'validation_curve.json');curve=curve if isinstance(curve,list)else[]
                live=read(directory/f'{group}_live.json');audit=read(directory/f'audits/{group}.json')
                step=checkpoint.get('updates',0);best=min((r['prompt_equal_error']for r in curve),default=None)
                metrics=[]
                if(run/'metrics.jsonl').exists():
                    for line in(run/'metrics.jsonl').read_text().splitlines()[-8:]:
                        try:metrics.append(json.loads(line))
                        except ValueError:pass
                timing=[r.get('execution_timing',{}).get('update_seconds')for r in metrics];timing=[x for x in timing if isinstance(x,(int,float))]
                speed=f'{statistics.median(timing):.1f}s/步'if timing else'等待计时'
                label='已完成'if(run/'completed.json').exists()else live.get('status','等待审计'if not audit.get('passed')else'准备启动')
                print(f" {group}  {label:18} {step:4}/2000最低预算  {speed:12} Val最佳 {best if best is not None else '—'}")
        if state.get('error'):print('\n错误：'+state['error'])
        print(f'\n{root}')
        if not a.watch:break
        time.sleep(5)


if __name__=='__main__':main()
