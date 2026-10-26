"""Readable paired baseline video report; no checkpoint selection."""
import html,json,statistics
from .setup import ROOT
from ..common import atomic_json

def main():
    cases=[json.loads((p.parent/'result.json').read_text()) for p in sorted((ROOT/'e2e_before').glob('*.seed0/completed.json'))]
    assert len(cases)==8,len(cases)
    summary={}
    for name in ['Tile15_before','Tile20_before']:
        values=[c['variants'][name] for c in cases]
        summary[name]={k:statistics.mean(v[k] for v in values) for k in ['weighted_keep','executed_attention_ratio']}
        summary[name].update({k:statistics.mean(v['quality_vs_dense']['sparse_frames_only'][k] for v in values) for k in ['lpips','psnr_db']})
        assert all(v['first_chunk_identical'] and v['sparse_layers']==list(range(1,30)) for v in values)
    atomic_json(ROOT/'before_summary.json',dict(cases=8,metrics=summary,stage='before_finetuning',quality_frames='last36',timing_measured=False))
    lines=['Opt20 全层微调前视频评估（8条，480×832，45帧）',
        '层1–29稀疏；层0及首chunk Dense。LPIPS/PSNR仅统计后36帧，与同种子Dense配对。',
        '以下结果来自微调前模型，不能据此判断微调效果。执行率仅统计稀疏attention区域，不代表整网FLOPs。',
        '方案           keep%   execute%   LPIPS↓   PSNR↑']
    for name,d in summary.items():lines.append(f"{name:14} {100*d['weighted_keep']:.3f} {100*d['executed_attention_ratio']:.3f} {d['lpips']:.6f} {d['psnr_db']:.3f}")
    lines+=['','prompt                 Tile15 LPIPS    Tile20 LPIPS']
    body=['<!doctype html><meta charset="utf-8"><title>Opt20 全层微调前视频</title>',
        '<style>body{font-family:sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;background:#17191d;color:#eee}video{width:100%;margin-bottom:24px}a{color:#8cf}td,th{padding:8px;text-align:left}</style>',
        '<h1>全层微调前：Dense / Tile15 / Tile20</h1><p>每条视频从左到右：Dense、实际约15%执行、实际约20%执行。层0与首chunk Dense；层1–29稀疏。以下不是微调后的结果。</p>',
        '<p>8条验证视频；45帧，480×832。LPIPS和PSNR与同种子Dense比较，仅统计后36帧。执行率只对应稀疏attention区域。</p>',
        '<a href="BEFORE_REPORT.txt">指标明细</a><table><tr><th>方案</th><th>保留率</th><th>执行率</th><th>LPIPS↓</th><th>PSNR↑</th></tr>']
    for name,d in summary.items():body.append(f"<tr><td>{name}</td><td>{100*d['weighted_keep']:.2f}%</td><td>{100*d['executed_attention_ratio']:.2f}%</td><td>{d['lpips']:.4f}</td><td>{d['psnr_db']:.2f}</td></tr>")
    body.append('</table>')
    for c in cases:
        pid=c['prompt_id'];a,b=[c['variants'][v]['quality_vs_dense']['sparse_frames_only']['lpips'] for v in ['Tile15_before','Tile20_before']]
        lines.append(f'{pid} {a:.6f} {b:.6f}')
        body.append(f'<h2>{html.escape(pid)}</h2><p>LPIPS：Tile15 {a:.4f} / Tile20 {b:.4f}</p><video controls preload="metadata" src="e2e_before/{pid}.seed0/Dense_Tile15_Tile20_before.mp4"></video>')
    (ROOT/'BEFORE_REPORT.txt').write_text('\n'.join(lines)+'\n')
    (ROOT/'before_videos.html').write_text('\n'.join(body))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
