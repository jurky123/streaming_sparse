import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'artifacts/comparison_20261113'
data=json.loads((root/'summary.json').read_text());rows=data['rows'];methods=['dense','ours_latest_summary','ours_latest_summary_tail','ours_latest_rope','ours_latest_rope_tail','ours_latest_cached','ours_latest_cached_tail','ours_latest_stable','ours_latest_optimized','ours_latest_fused','ours_latest_fast','ours','dasa','pisa','solattn','svgear','lightforcing','forcingkv'];labels={'ours_latest_summary':'Ours: fused summary','ours_latest_summary_tail':'Ours: summary + fast tail','dense':'Dense','ours_latest_rope':'Ours: fused RoPE + cache', 'ours_latest_rope_tail':'Ours: fused RoPE + fast tail', 'ours_latest_cached':'Ours: cached execution', 'ours_latest_cached_tail':'Ours: cached + faster tail', 'ours_latest_stable':'Ours: previous execution', 'ours_latest_optimized':'Ours: faster tail candidate', 'ours_latest_fused':'Ours: previous fused route', 'ours_latest_fast':'Ours: initial kernels','ours':'Ours: FP32 reference','dasa':'DASA','pisa':'PISA (padded Q)','solattn':'Sol-Attn (padded Q)','svgear':'SVG-EAR','lightforcing':'LightForcing','forcingkv':'Forcing-KV'}
fig,axes=plt.subplots(2,3,figsize=(15,10.5),constrained_layout=True)
for i,case in enumerate(['horse_field','traffic_crossing']):
 rr={r['method']:r for r in rows if r['case']==case};ms=[m for m in methods if m in rr];y=np.arange(len(ms))
 for j,(key,title) in enumerate([('total_seconds','Generation + VAE (seconds, log scale)\nLower is better'),('early_lpips','Early LPIPS vs Dense\nLower is closer'),('late_lpips','Late LPIPS vs Dense\nLower is closer')]):
  ax=axes[i,j];vals=[rr[m][key] if rr[m][key] is not None else 0 for m in ms];ax.barh(y,vals,color=['#2563eb' if m=='ours_latest_rope' else '#94a3b8' for m in ms]);ax.invert_yaxis();ax.set_yticks(y,[labels[m] for m in ms] if j==0 else ['']*len(ms));ax.set_title(case+'\n'+title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
  if j==0:ax.set_xscale('log');ax.set_xlim(min(vals)*.65,max(vals)*1.7)
  if j>0:ax.set_xlim(0,max(vals)*1.2)
  for yy,v in zip(y,vals):ax.text(v,yy,f' {v:.2f}' if j==0 else f' {v:.3f}',va='center',fontsize=8)
fig.suptitle('A100 | 512 × 768 | 3 latent frames/chunk | 15-frame history cap | 480 output frames\nSame Self-Forcing weights and saved inputs; native compression policies; two timed repetitions'+(' | INCOMPLETE' if not data['complete'] else ''),fontsize=11)
fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)
