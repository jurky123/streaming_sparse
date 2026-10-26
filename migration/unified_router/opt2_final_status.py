"""Readable status for the selected, numerically guarded adapter."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / 'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024/optimization2'
    while True:
        summary = json.loads((root / 'summary.json').read_text())
        checks = json.loads((root / 'verification.json').read_text())
        lines = ['SlimAV 第二轮热点优化 — 最终 Guarded 版',
                 '状态：已完成 | 逐 query 独立路由 | TF32 关闭', '',
                 '方案                 生成秒     含VAE总秒']
        for name, row in summary['speed'].items():
            lines.append(f"{name:<20} {row['generation_s']:>7.2f}     {row['total_s']:>7.2f}")
        lines += ['', f"完整 latent/RGB 逐 bit 一致：{checks['complete_latent_and_rgb_bitwise_checks']}/16 组（对上一轮 Fast1）",
                  f"冻结文件校验：{checks['sealed_files_verified']} 份通过",
                  '最终 profile：已完成；路由 21.60 → 10.64 ms / 代表层',
                  '推荐入口：migration.unified_router.opt2_guarded.load_guarded',
                  '前期无保护 Opt2 与 strict 分支未采用。',
                  f"报告：{root / 'REPORT.md'}"]
        print(('' if args.once else '\033[2J\033[H') + '\n'.join(lines), flush=True)
        if args.once:
            return
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    main()
