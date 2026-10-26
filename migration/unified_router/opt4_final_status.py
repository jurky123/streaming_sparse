"""Read final timings and the post-fusion full-video regression status."""
import argparse
import time
from pathlib import Path
from .opt4.status import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization4'
    while True:
        state=read(root/'fused_status.json')
        lines=['SlimAV IO-aware 路由 — 最终融合版',f"状态：{state.get('status','准备')} | 阶段：{state.get('stage','—')}",
            '逐query独立路由；省去全页content和已选权重复制；原主干和head未重训。','',
            f"{'方案':<22} {'重复':>5} {'生成/总秒':>17}"]
        for folder in sorted((root/'fused_benchmark').glob('*')):
            rows=read(folder/'timings.json').get('records',[])[1:]
            avg=lambda k:sum(r[k]for r in rows)/len(rows)if rows else 0.
            lines.append(f"{folder.name:<22} {len(rows):>3}/3 {avg('generation_seconds'):>7.2f}/{avg('total_seconds'):<7.2f}")
        checks=[]
        for i in range(8):
            value=read(root/f'fused_validation/results_rank{i}.json')
            if isinstance(value,list):checks+=value
        passed=sum(r['latent_bit_identical']and r['rgb_bit_identical']for r in checks)
        lines+=['',f"质量回归：{len(list((root/'quality').glob('*/result.json')))}/8 prompt，32组生成",
            f"最终融合保持上述视频：{passed}/{len(checks)} 已检，目标32组latent/RGB逐bit一致"]
        for name in ['hybrid','fp16']:
            lines.append(f"最终{name} profile：{read(root/f'profile_fused_{name}/status.json').get('status','等待')}")
        if (root/'REPORT.md').exists():lines.append('报告：'+str(root/'REPORT.md'))
        print((''if a.once else '\033[2J\033[H')+'\n'.join(lines),flush=True)
        if a.once:return
        try:time.sleep(5)
        except KeyboardInterrupt:return


if __name__=='__main__':main()
