"""Summarize decoded, paired full-network rollouts and collect viewing links."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
import numpy as np

from .common import config,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args()
    root,m=config(a.directory);manifest=json.loads((root/'e2e_manifest.json').read_text())
    rows=[]
    for pid in manifest['prompt_ids']:
        for seed in manifest['generation_seeds']:
            path=root/f'videos/{pid}.seed{seed}/result.json'
            row=json.loads(path.read_text());assert row['first_dense_frames_identical'];rows.append(row)
    modes=manifest['modes'][1:];summary={};csvrows=[]
    for mode in modes:
        values=[r['metrics'][mode]for r in rows]
        mse=statistics.mean(v['quality']['sparse_frames_only']['mse_rgb01']for v in values)
        summary[mode]=dict(psnr_db=-10*math.log10(mse)if mse else None,
            lpips=statistics.mean(v['quality']['sparse_frames_only']['lpips']for v in values),
            macro_keep=statistics.mean(v['budget']['macro_keep']for v in values),
            token_weighted_keep=statistics.mean(v['budget']['token_weighted_keep']for v in values),
            history_keep=statistics.mean(v['budget']['history_keep']for v in values),
            current_keep=statistics.mean(v['budget']['current_keep']for v in values),
            final_chunk_latent_nmse=statistics.mean(v['latent_chunks'][-1]['nmse']for v in values))
        for r,v in zip(rows,values):
            q=v['quality']['sparse_frames_only']
            csvrows.append(dict(prompt_id=r['prompt_id'],seed=r['generation_seed'],mode=mode,
                psnr_db=q['psnr_db'],lpips=q['lpips'],mse_rgb01=q['mse_rgb01'],**v['budget']))
    # Pair across the same prompts; both generation seeds stay in each cluster.
    deltas=[]
    for pid in manifest['prompt_ids']:
        cohort=[r for r in rows if r['prompt_id']==pid]
        deltas.append(statistics.mean(r['metrics']['V32-Set']['quality']['sparse_frames_only']['lpips']-
            r['metrics']['K32-Set']['quality']['sparse_frames_only']['lpips']for r in cohort))
    rng=np.random.default_rng(20261024);d=np.asarray(deltas)
    bootstrap=d[rng.integers(0,len(d),(8000,len(d)))].mean(1)
    paired=dict(V_minus_K_LPIPS=float(d.mean()),bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
                bootstrap_unit='prompt, both generation seeds clustered',prompts=len(d))
    out=root/'video_review';out.mkdir(exist_ok=True)
    atomic_json(out/'summary.json',dict(completed=True,cases=len(rows),manifest=manifest,modes=summary,
        paired=paired,metric_scope='Frames9..44, uncompressed RGB against same-noise Dense; consistency, not absolute perceptual quality.',
        first9_frames_identical=True,layer0_dense=True,sparse_layers=list(range(1,30)),test_used=False))
    with(out/'metrics.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    text=['# Slim-AV v0.10：29层端到端视频测试','',
        f'第0层Dense，第1–29层全部12Heads稀疏。首chunk Dense；后续首chunk KV作为普通历史候选。历史原型P8/P4/P2、当前P16。β={manifest["beta"]}，训练集校准约15%总KV。',
        f'固定训练seed0的实际注意力NMSE最佳权重；{len(manifest["prompt_ids"])}个注册Val提示词×两个生成种子，共{len(rows)}组。Test未使用。',
        '每组四个原始latent chunks，解码45帧、480×832、16fps。首9帧逐位一致；主质量指标只比较后36帧，来自压缩前RGB。',
        '', '| 方案 | PSNR ↑ | LPIPS ↓ | KV保留率（宏平均） | KV保留率（token加权） |', '|---|---:|---:|---:|---:|']
    for mode,v in summary.items():
        text.append(f'| {mode} | {v["psnr_db"]:.3f} | {v["lpips"]:.6f} | {v["macro_keep"]:.2%} | {v["token_weighted_keep"]:.2%} |')
    text += ['',f'V−K配对LPIPS差：{paired["V_minus_K_LPIPS"]:.6f}；按Prompt聚类bootstrap95%区间：{paired["bootstrap95"]}。负数表示V更接近Dense。',
        '保留率仅统计后续chunk的29个稀疏层，历史和当前合计；第0层、首chunk和Dense上下文写入未计入该比率。',
        '这些指标衡量与同噪声Dense生成的接近程度，不代表绝对视频质量。不同方案实际保留率可能略有不同；本测试不作推理加速结论。',
        '', '## 全部视频', '']
    cards=[]
    for i,row in enumerate(rows,1):
        label=f'{i:02d} · {row["prompt_id"]} · seed{row["generation_seed"]}'
        main=Path(row['comparison']).relative_to(root);baseline=Path(row['baseline_comparison']).relative_to(root)
        text.append(f'- {label}：[Dense / K32 / V32](../{main})；[Dense / U-Sol / V32](../{baseline})')
        cards.append(f'<section><h2>{html.escape(label)}</h2><p>左Dense · 中K32 · 右V32</p><video controls preload="none" src="../{main}"></video><details><summary>查看U-Sol基线对照</summary><video controls preload="none" src="../{baseline}"></video></details></section>')
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Slim-AV v0.10 端到端对照</title><style>body{font:16px/1.7 system-ui;max-width:1440px;margin:24px auto;padding:0 20px;background:#f4f6f9;color:#182330}section{background:white;padding:20px;margin:20px 0;border-radius:12px}video{width:100%;background:#111}a{color:#165ca6}</style>'
        '<h1>第0层Dense · 第1–29层稀疏</h1>'
        f'<p>{len(rows)}组同提示词、同噪声闭环视频；首chunk Dense，后续约15%总KV。全部案例按固定顺序展示。</p>'
        '<p><a href="REPORT.md">评估报告</a> · <a href="metrics.csv">逐视频指标</a> · <a href="quality_curves.png">质量曲线</a></p>'
        +''.join(cards)+'</html>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for mode in modes:
        lp=np.array([[v['lpips']for v in r['metrics'][mode]['quality']['per_frame']]for r in rows]).mean(0)
        latent=np.array([[v['nmse']for v in r['metrics'][mode]['latent_chunks']]for r in rows]).mean(0)
        axes[0].plot(range(9,45),lp[9:],label=mode)
        axes[1].plot(range(4),latent,marker='o',label=mode)
    axes[0].set(xlabel='Decoded frame (first9 Dense frames excluded)',ylabel='LPIPS to Dense (lower is better)')
    axes[1].set(xlabel='Latent chunk',ylabel='Closed-loop latent NMSE to Dense')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'quality_curves.png',dpi=160);fig.savefig(out/'quality_curves.pdf');plt.close(fig)
    print(json.dumps(dict(cases=len(rows),summary=summary,paired=paired,report=str(out/'REPORT.md'))),flush=True)


if __name__=='__main__':main()
