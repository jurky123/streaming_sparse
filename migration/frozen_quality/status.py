"""Dependency-free progress view for the frozen quality assessment."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    try:return json.loads(path.read_text())
    except(FileNotFoundError,ValueError):return{}


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    root=Path(read(ROOT/'migration/frozen_quality_active.json')['directory'])
    while True:
        manifest=read(root/'manifest.json');offline=0;videos=0;lines=[]
        for rank in range(8):
            state=read(root/f'rank{rank}/status.json');phase=state.get('status','等待启动');mode=state.get('mode','—')
            done=sum(len(read(path).get('rows',[]))for path in(root/f'rank{rank}').glob('*_offline.json'))
            if phase=='offline'and not(root/f'rank{rank}/{mode}_offline.json').exists():done+=state.get('offline_completed',0)
            done=min(done,144);offline+=done
            vr=read(root/f'rank{rank}/video_rows.json');count=len(vr)if isinstance(vr,list)else 0;videos+=count
            lines.append(f"GPU{rank}  {phase:10} {mode:8}  离线 {done:3}/144  视频 {count}/2")
        if args.watch:print('\033[H\033[2J',end='')
        print(f'冻结骨干质量评估 | 离线 {offline}/1152 | 视频 {videos}/16')
        print('训练保持暂停于1039步；此处为评估任务。')
        print('\n'.join(lines));print(f'结果目录：{root}')
        if not args.watch:break
        time.sleep(5)


if __name__=='__main__':main()
