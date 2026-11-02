"""Rebuild concise Markdown from immutable quality gates and completed trials."""
import json,statistics
from pathlib import Path
BASE=Path('artifacts');DEST=BASE/'compensation_kernel_opt_20261030/acceptance_progress.zh.md'
def main():
    lines=['# Kernel 优化验收进度','','前段质量：RGB [21,101)。速度：完整30秒生成含VAE，未含加载、预热、指标、编码。只统计已写入原子结果的单位。', '', '| 实现 | 场景 | 质量门槛 | 全片latent逐位一致于原补偿 | LPIPS | SSIM | PSNR |','|---|---|---|---|---:|---:|---:|']
    roots=[('split','compensation_optimized_accept_20261030'),('fast','compensation_fast_accept_20261030'),('tf32pack','compensation_exact_accept_20261030'),('stable','compensation_stable_accept_20261030')]
    for backend,name in roots:
        for p in sorted((BASE/name).glob('*/gate.json')):
            r=json.loads(p.read_text());q=r['current'];lines.append(f"| {backend} | {p.parent.name} | {r['passed']} | {r['full_latents_exact_to_raw']} | {q['lpips']:.5f} | {q['ssim']:.5f} | {q['psnr']:.3f} |")
    lines+=['','| 实验 | 场景 | 方法 | 完成重复数 | 完整耗时中位秒 | 最小—最大 | 所有重复无同卡外部进程采样 |','|---|---|---|---:|---:|---|---|']
    for backend,name in roots:
        for folder in sorted((BASE/name).glob('*')):
            if not folder.is_dir():continue
            methods=sorted({p.name.split('.rep')[0] for p in folder.glob('*.rep*.json')})
            for m in methods:
                rows=[json.loads(p.read_text()) for p in folder.glob(m+'.rep*.json')];ts=[r['total_generation_vae_seconds'] for r in rows]
                clean=all(r['telemetry']['no_observed_foreign_processes'] for r in rows)
                lines.append(f'| {backend} | {folder.name} | {m} | {len(rows)} | {statistics.median(ts):.3f} | {min(ts):.3f}—{max(ts):.3f} | {clean} |')
    lines+=['','注意：共享负载计时不作为稳定加速比验收；单kernel无外部负载的数字与这些视频计时不能直接相除。split/fast质量回归失败，不纳入速度晋级。']
    DEST.write_text('\n'.join(lines)+'\n');print(DEST)
if __name__=='__main__':main()
