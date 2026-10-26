"""Human-readable optimization progress; no CUDA/model dependencies."""
import argparse,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    out=ROOT/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization'
    while True:
        state=read(out/'status.json');lines=['SlimAV v10 优化进度',f"状态: {state.get('status','准备中')} | 阶段: {state.get('stage','—')}",
            '',f"{'方案':<22} {'状态':<22} {'计时次数':>6} {'生成/总秒':>15}"]
        for d in sorted((out/'final_benchmark').glob('*')):
            s=read(d/'status.json');timing=read(d/'timings.json');rows=timing.get('records',[])[1:]
            mean=lambda key:sum(r[key]for r in rows)/len(rows)if rows else 0
            lines.append(f"{d.name:<22} {s.get('status','等待'):<22} {len(rows):>5}/3 {mean('generation_seconds'):>6.2f}/{mean('total_seconds'):<6.2f}")
        lines+=['','Val 视频质量回归（固定 8 个 prompt，seed0）']
        for rank in range(8):
            s=read(out/f'quality_seed0/rank{rank}.json')
            if s:lines.append(f"GPU{rank}: {s.get('status')} | {s.get('variant','')} | {s.get('prompt','')}")
        done=len(list((out/'quality_seed0').glob('*/result.json')))
        lines.append(f'质量完成: {done}/8 | 输出: {out}')
        lines.append('最终模块 profile: '+' | '.join(f"Q{stride}: {read(out/f'profile_q{stride}/status.json').get('status','等待')}"for stride in [1,16]))
        print(('\033[2J\033[H'if not a.once else '')+'\n'.join(lines),flush=True)
        if a.once:return
        try:time.sleep(5)
        except KeyboardInterrupt:return

if __name__=='__main__':main()
