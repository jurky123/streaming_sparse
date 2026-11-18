"""Report paired Dense fidelity changes, retaining the scene and seed structure."""
import argparse
import json
import statistics as st
from .validate_single_av import PROMPTS,TIERS,REPORT

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group-direction',action='store_true')
    grouping=parser.parse_args().group_direction
    folder=REPORT.parent.parent/'group_direction_v1/validation' if grouping else REPORT
    label='小节点分组融合' if grouping else '精算 AV 单项与双项'
    rows=[]
    for prompt in PROMPTS:
        for seed in range(703,707):
            group=json.loads((folder/f'{prompt}_{seed}.json').read_text())
            assert sorted(r['tier'] for r in group)==sorted(TIERS)
            assert all(r['prompt']==prompt and r['seed']==seed for r in group)
            rows.extend(group)
    lines=[f'# {label}：跨场景验证','',
           '冻结的 5 个 prompt × seed703–706，每组同卡重新生成 Dense 和三个档位的两种实现，每组 7 条轨迹（分组实验复用同卡的 Dense 与旧实现）。当前融合 QK 基线；9 chunks，105 帧。质量窗口 [21,101)，LPIPS/SSIM 使用 256×384，PSNR 使用原分辨率。所有差值为新实现减旧实现。',
           '', '| 档位 | 原 LPIPS | 新 LPIPS | ΔLPIPS | 改善/持平/变差 | ΔSSIM | ΔPSNR/dB | 新旧间 LPIPS |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def diff(r,k):return r['new'][k]-r['old'][k]
    for tier in TIERS:
        rs=[r for r in rows if r['tier']==tier]
        ds=[diff(r,'lpips') for r in rs]
        a,b=[st.mean(r[m]['lpips'] for r in rs) for m in ('old','new')]
        lines.append(f"| {tier} | {a:.5f} | {b:.5f} | {st.mean(ds):+.5f} | {sum(x<0 for x in ds)}/{ds.count(0)}/{sum(x>0 for x in ds)} | {st.mean(diff(r,'ssim') for r in rs):+.5f} | {st.mean(diff(r,'psnr') for r in rs):+.3f} | {st.mean(r['paired']['lpips'] for r in rs):.5f} |")
    lines+=['','| 场景平均 ΔLPIPS | base | intersection | step |','|---|---:|---:|---:|']
    for p in PROMPTS:
        means=[st.mean(diff(r,'lpips') for r in rows if r['tier']==t and r['prompt']==p) for t in TIERS]
        lines.append('| '+p+' | '+' | '.join(f'{m:+.5f}' for m in means)+' |')
    lines+=['','逐样本 ΔLPIPS 的最小值、中位数与最大值（20 个配对）：','']
    for t in TIERS:
        ds=[diff(r,'lpips') for r in rows if r['tier']==t]
        lines.append(f'- {t}：{min(ds):+.5f} / {st.median(ds):+.5f} / {max(ds):+.5f}。')
    lines+=['','各档相对 Dense 的 LPIPS 变差最大的样本：','']
    for t in TIERS:
        r=max((r for r in rows if r['tier']==t),key=lambda r:diff(r,'lpips'))
        lines.append(f"- {t}：{r['prompt']} / {r['seed']}，{r['old']['lpips']:.5f} → {r['new']['lpips']:.5f}（{diff(r,'lpips'):+.5f}）。")
    (folder/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
