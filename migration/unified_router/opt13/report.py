"""Aggregate paired results; keep raw data and a local comparison gallery."""
import json,statistics,html
from pathlib import Path
root=Path(__file__).resolve().parents[3]/'streaming_sparse/v10/experiments/opt13_compact_20261025'
mean=statistics.mean
folders=sorted((root/'quality').glob('*.seed0'))
assert len(folders)==8 and all((f/'timing_summary.json').exists()for f in folders)
names=['Reference','Vote2','Vote4','Vote8'];rows={};per_prompt=[]
for f in folders:
 timing=json.loads((f/'timing_summary.json').read_text());metrics={n:json.loads((f/(n+'.json')).read_text())for n in names}
 per_prompt.append(dict(prompt=f.name,timing=timing,metrics=metrics))
for n in ['Dense']+names:
 row={'generation_seconds':mean(p['timing'][n]for p in per_prompt),'timing_prompts':8,'timed_repeats_per_prompt':3}
 if n!='Dense':
  row.update(lpips=mean(p['metrics'][n]['quality_vs_dense']['sparse_frames_only']['lpips']for p in per_prompt),psnr_db=mean(p['metrics'][n]['quality_vs_dense']['sparse_frames_only']['psnr_db']for p in per_prompt),weighted_keep=mean(p['metrics'][n]['weighted_keep']for p in per_prompt),lpips_wins_vs_reference=sum(p['metrics'][n]['quality_vs_dense']['sparse_frames_only']['lpips']<p['metrics']['Reference']['quality_vs_dense']['sparse_frames_only']['lpips']for p in per_prompt))
 rows[n]=row
for n,row in rows.items():
 row['speedup_vs_reference']=rows['Reference']['generation_seconds']/row['generation_seconds'];row['speedup_vs_dense']=rows['Dense']['generation_seconds']/row['generation_seconds']
 if n not in ['Dense','Reference']:row['relative_lpips_change']=row['lpips']/rows['Reference']['lpips']-1
micro={f.stem:json.loads(f.read_text())for f in sorted(root.glob('vote_probe_layer*.json'))}
result=dict(protocol='8 native Val prompts, seed0, 480x832, 45 frames; quality on 36 sparse frames; same-GPU alternating warm generation timing, median of 3 per prompt then mean; VAE excluded',baseline='Opt11 Compact strong-stage best, same parameters for all vote variants',summary=rows,per_prompt=per_prompt,micro=micro,default_variant='Reference',candidate_recommendation='Vote2 for minimal average quality change; Vote4 for a larger speed/quality tradeoff; Vote8 not adopted',limitations=['8 prompts, one seed; no retraining','Voting errors can accumulate; average LPIPS does not establish equivalence','Default remains Reference; no prior experiment artifacts modified'])
result['phase_spans']={f.stem:json.loads(f.read_text())for f in root.glob('phases_*.json')}
result['kernel_profiles']={name:json.loads((root/f'profile_{name}_step3/kernel_attribution.json').read_text())['modules']for name in ['Reference','Vote4','FusedVote4']}
result['fused_micro']={f.stem:json.loads(f.read_text())for f in root.glob('fused_vote_probe_layer*.json')}
result['fused_status']='Experimental: exact against separate voting in 18 real-block cases (6 blocks x 3 thresholds); one full-trajectory repeat/profile; no 8-prompt uninstrumented paired benchmark'
(root/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['Opt13：Compact query tile 投票实验','基线：Opt11 Compact；不使用 Skip0 / FP16 PV / Q64。','32 query 一组，按 KV page 投票，仍保留各 query 自己的 mask。低票过滤后若 query 为空，恢复原选择。尾组按有效 query 数缩放门槛。','验证：8 条 Val，seed0，45帧，480×832；LPIPS/PSNR 仅统计后36帧。首chunk与Dense完全相同。','测速：每条视频同卡交替执行，额外预热后3次，取中位数再跨8条求均值；含context重建，不含VAE。','所有20次复跑/视频与同轮质量评估latent逐位一致。','', '方案        生成秒    vs原版倍速   LPIPS      保留率']
for n,r in rows.items():lines.append(f"{n:10s} {r['generation_seconds']:.4f}    {r['speedup_vs_reference']:.4f}       {r.get('lpips',0):.6f}   {r.get('weighted_keep',1):.2%}")
lines+=['','Vote2、Vote4 为候选；Vote8 的质量损失较大，不采用。默认仍为 Reference。','候选选择并行化另有两种实现：单CTA多池显著变慢；分CTA多池只在部分短历史形状有小收益，长历史更慢，未集成。','完整证据见 report.json；视频见 index.html。']
lines+=['','完整轨迹分段（单prompt，独立插桩测量，CUDA event区间可包含CPU提交间隙）：','原版 sparse attention 合计482.7ms；4票373.7ms。路由（含投票）800.3→824.0ms，净节省约85ms。','首chunk Dense约685ms，历史重建约1130ms；这些开销未被投票减少。','分段计时有嵌套，不能把 later_chunks_forward 与内部路由/attention 时间再次相加。','融合投票+union在18个真实block/门槛组合上与分离实现逐位一致；另通过18个空query回退/尾tile合成案例。','融合版仅作实验候选，未做8条视频的独立成对测速；不能将单次插桩时间当正式端到端结果。','代表block路由+投票约2.61ms，attention约1.50ms；接下来主要热点是路由及未受影响的主干/历史重建。']
(root/'report.txt').write_text('\n'.join(lines)+'\n')
head='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Opt13 投票实验</title><style>body{max-width:1500px;margin:28px auto;padding:0 20px;font:16px system-ui;background:#111827;color:#e5e7eb}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #475569}video{width:100%;background:#000}article{margin:32px 0}a{color:#93c5fd}</style><h1>Opt13：query tile 投票实验</h1><p>视频从左到右：Dense / 原 Compact / 4票 / 8票。横向四屏可全屏查看。</p><p>LPIPS 越低越好；仅统计后36帧。生成时间不含VAE。原实验与checkpoint未修改。</p><table><tr><th>方案</th><th>生成秒</th><th>相对原版倍速</th><th>LPIPS</th><th>保留率</th></tr>'
for n,r in rows.items():head+=f"<tr><td>{n}</td><td>{r['generation_seconds']:.4f}</td><td>{r['speedup_vs_reference']:.3f}</td><td>{r.get('lpips',0):.6f}</td><td>{r.get('weighted_keep',1):.2%}</td></tr>"
head+='</table><p>2票平均质量接近原版；4票有小幅损失；8票不采用。以上仅覆盖8条样本、单个seed，未再训练。</p>'
for p in per_prompt:
 f=p['prompt'];vals=' / '.join(f"{n}: {p['metrics'][n]['quality_vs_dense']['sparse_frames_only']['lpips']:.4f}"for n in names)
 head+=f'<article><h2>{html.escape(f)}</h2><p>LPIPS：{vals}</p><video controls preload="none" src="quality/{f}/Dense_Reference_Vote4_Vote8.mp4"></video></article>'
(root/'index.html').write_text(head+'</html>')
print('\n'.join(lines))
