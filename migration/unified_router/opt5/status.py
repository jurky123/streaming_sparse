"""Human-readable progress without importing torch."""
import argparse,json,time
from pathlib import Path

def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization5'
    while True:
        if not a.once:print('\033[2J\033[H',end='')
        state=read(root/'status.json');print('SlimAV Opt5 · FP16 路由减少显存中间量')
        print('阶段:',state.get('stage','内核探索 / 正确性检查'),' 状态:',state.get('status','运行中'))
        probes=[read(f)for f in (root/'probes').glob('*.json')if not f.name.endswith('.status.json')]
        print('已完成微基准:',len(probes))
        checks=read(root/'checks.json');print('跨层 / 历史 / K-V 检查:',checks.get('status','待开始'),len(checks.get('rows',[])),'/ 18')
        for folder in sorted((root/'benchmark').glob('*')):
            s=read(folder/'status.json');data=read(folder/'timings.json');rows=data.get('records',[])[1:]
            timing=f" · 生成 {sum(r['generation_seconds']for r in rows)/len(rows):.3f}s"if rows else ''
            print('速度',folder.name,s.get('status','等待'),f'{len(rows)}/3'+timing)
        rows=list((root/'quality').glob('*.seed0/Opt5-*.json'));print('视频质量:',len(rows),'/ 16')
        for f in sorted((root/'quality').glob('rank*.json')):
            s=read(f);print(' ',f.stem,s.get('status'),s.get('prompt',''),s.get('mode',''))
        print('结果目录:',root)
        if a.once:return
        time.sleep(5)

if __name__=='__main__':main()
