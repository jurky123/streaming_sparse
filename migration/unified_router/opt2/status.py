"""Lightweight second-stage optimization dashboard."""
import argparse,json,time
from pathlib import Path
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization2'
    while True:
        s=read(root/'status.json');lines=['SlimAV 第二轮热点优化 — 逐 query 独立路由',f"状态: {s.get('status','准备')} | 阶段: {s.get('stage','—')}",'',f"{'方案':<22} {'计时':>5} {'生成/总秒':>16}  状态"]
        for d in sorted((root/'benchmark').glob('*')):
            state=read(d/'status.json');rr=read(d/'timings.json').get('records',[])[1:]
            avg=lambda k:sum(r[k]for r in rr)/len(rr)if rr else 0
            lines.append(f"{d.name:<22} {len(rr):>3}/3 {avg('generation_seconds'):>7.2f}/{avg('total_seconds'):<7.2f} {state.get('status','等待')}")
        lines+=['',f"质量回归: {len(list((root/'quality').glob('*/result.json')))}/8 prompt"]
        for i in range(8):
            state=read(root/f'quality/rank{i}.json')
            if state:lines.append(f"GPU{i}: {state.get('status')} {state.get('variant','')} {state.get('prompt','')}")
        lines.append('最终 profile: '+read(root/'profile_q1/status.json').get('status','等待'))
        print(('\033[2J\033[H'if not a.once else '')+'\n'.join(lines),flush=True)
        if a.once:return
        try:time.sleep(5)
        except KeyboardInterrupt:return
if __name__=='__main__':main()
