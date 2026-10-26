"""Read-only compact-route experiment dashboard."""
import argparse
import json
import time
from pathlib import Path


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization4'
    while True:
        state=read(root/'status.json');lines=['SlimAV IO-aware 路由实验 — 全页 content 不落显存',
            f"状态：{state.get('status','准备')} | 阶段：{state.get('stage','微基准/正确性检查')}",
            'Hybrid：高精度打分＋FP16统计；FP16：打分也使用FP16；逐query独立选页。','',
            f"边界/多层检查：{read(root/'checks.json').get('status','等待')}",
            f"{'方案':<22} {'重复':>5} {'生成/总秒':>17}"]
        for folder in sorted((root/'benchmark').glob('*')):
            rows=read(folder/'timings.json').get('records',[])[1:]
            avg=lambda k:sum(r[k]for r in rows)/len(rows)if rows else 0.
            lines.append(f"{folder.name:<22} {len(rows):>3}/3 {avg('generation_seconds'):>7.2f}/{avg('total_seconds'):<7.2f}")
        lines+=['',f"视频回归：{len(list((root/'quality').glob('*/result.json')))}/8 prompt，共32组生成"]
        for rank in range(8):
            st=read(root/f'quality/rank{rank}.json')
            if st:lines.append(f"GPU{rank}: {st.get('status')} {st.get('variant','')} {st.get('mode','')} {st.get('prompt','')}")
        print((''if a.once else '\033[2J\033[H')+'\n'.join(lines),flush=True)
        if a.once:return
        try:time.sleep(5)
        except KeyboardInterrupt:return


if __name__=='__main__':main()
