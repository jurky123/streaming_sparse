"""Aggregate frozen cross-prompt validation; no policy selection or retuning."""
import csv,json,statistics,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .intersection_validation import OUT,CASES,SEEDS,METHODS
LABELS=dict(dense='Dense',original='原交集',search_mean='搜索均值最优',search_balanced='搜索双seed均衡',top_layers='按层选择',top_heads='按head选择',top_steps='按step选择')
def main():
 assert json.loads((OUT/'status.json').read_text())['state']=='complete'
 data={(c,s,m):json.loads((OUT/'videos'/f'{c}_s{s}'/m/'result.json').read_text()) for c in CASES for s in SEEDS for m in ['dense']+METHODS}
 rows=[]
 for (c,s,m),d in data.items():
  q=d['quality'];r=dict(case=c,seed=s,method=m,gpu=d['gpu'],generation_seconds=d['generation_seconds'],decode_seconds=d['decode_seconds'])
  if q:
   r.update({k:q[k] for k in ['lpips','ssim','psnr']});r['lpips_delta_vs_original']=q['lpips']-data[c,s,'original']['quality']['lpips'];r['early_lpips']=statistics.mean(q['per_frame']['lpips'][:40]);r['late_lpips']=statistics.mean(q['per_frame']['lpips'][40:])
  rows.append(r)
 for c in CASES:
  for s in SEEDS:assert len({data[c,s,m]['gpu'] for m in ['dense']+METHODS})==1
 assert all(d['counters'][2]==0 and d['static_route'] for d in data.values())
 def aggregate(rs):
  result=[]
  for m in ['dense']+METHODS:
   selected=[r for r in rs if r['method']==m];x=dict(method=m,n=len(selected))
   for k in ['generation_seconds','decode_seconds','lpips','ssim','psnr','early_lpips','late_lpips','lpips_delta_vs_original']:
    if k in selected[0]:x[k]=statistics.mean(r[k] for r in selected)
   if m!='dense':
    ds=[r['lpips_delta_vs_original'] for r in selected];x.update(wins=sum(d < -1e-6 for d in ds),ties=sum(abs(d)<=1e-6 for d in ds),losses=sum(d>1e-6 for d in ds))
   result.append(x)
  for x in result:x['speedup_vs_dense']=result[0]['generation_seconds']/x['generation_seconds'];x['speedup_vs_original']=result[1]['generation_seconds']/x['generation_seconds']
  return result
 table=aggregate(rows);bycase={c:aggregate([r for r in rows if r['case']==c]) for c in CASES}
 def csvwrite(name,rs):
  keys=list(dict.fromkeys(k for r in rs for k in r))
  with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
 csvwrite('paired_results.csv',rows);csvwrite('summary.csv',table);csvwrite('by_prompt.csv',[dict(case=c,**r) for c,rs in bycase.items() for r in rs])
 (OUT/'summary.json').write_text(json.dumps(dict(table=table,by_prompt=bycase,audit=dict(videos=len(data),groups=20,same_gpu_per_group=True,nonfinite_scores=0)),indent=2)+'\n')
 lines=['# 交集预算搜索：5 prompt × 4 seed 冻结验证','', '搜索在 city_walk、seed701/702 上完成，341个不同配置、682个视频。此处固定两版搜索结果，与原交集及按层/head/step选择比较；验证结果不用于重新调参。','', '验证：kitchen_cook、forest_run、portrait_turn、reef_fish、tennis_match；seed703–706。共140视频，8个常驻GPU进程。9 chunks、105帧、768×512、16FPS、history360、tile72、static_route=True。同一prompt/seed的7种方法在同一GPU上执行，Dense仅生成解码一次，稀疏方法顺序轮换。','', 'LPIPS(AlexNet)/SSIM在256×384计算，PSNR使用原分辨率；指标窗口为帧[21,101)，前后段分别[21,61)、[61,101)。均为相对该样本Dense的保真度。prompt和seed未参加这次策略搜索，不代表未参加模型训练。','', '## 汇总','', '| 方法 | 生成/s↓ | 相对Dense加速↑ | LPIPS↓ | 前段LPIPS↓ | 后段LPIPS↓ | SSIM↑ | PSNR/dB↑ | 对原交集胜/平/负 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in table:
  cells=[LABELS[r['method']],f"{r['generation_seconds']:.3f}",f"{r['speedup_vs_dense']:.3f}×"]+[f"{r[k]:.5f}" if k in r else '—' for k in ['lpips','early_lpips','late_lpips','ssim','psnr']]+[f"{r['wins']}/{r['ties']}/{r['losses']}" if 'wins'in r else '—'];lines.append('| '+' | '.join(cells)+' |')
 lines+=['','耗时为完成基础/3-head/12-head路由形状预热后，每样本每方法一次生成；排除权重加载、VAE、指标和文件IO。表中速度比为均值之比。这是跨场景配对速度记录，不替代搜索报告中同卡三次重复的正式计时。所有方法共用实现，仅改变Full_Pset执行范围；不是kernel提速。','', '## 两版搜索结果的迁移','']
 original=table[1]
 for m in ['search_mean','search_balanced']:
  r=next(x for x in table if x['method']==m);delta=r['lpips']-original['lpips'];lines.append(f"- {LABELS[m]}：LPIPS相对原交集变化 {delta:+.6f}（{100*delta/original['lpips']:+.3f}%），{r['wins']}/20配对改善；生成加速{r['speedup_vs_original']:.4f}×。")
 lines+=['', '两版搜索配置的跨场景均值均未超过原交集，搜索集收益没有迁移到本次验证。因此保留原交集作为低预算方案，不用搜索结果替换默认。按step选择的平均LPIPS最低，但比原交集多耗时约0.662s（8.53%）；它使用25%修正预算，原交集仅1.51%，属于不同成本点。此处报告观察结果，不作统计显著性保证。', '', '## 逐场景LPIPS（各4seed）','','| 场景 | '+' | '.join(LABELS[m] for m in METHODS)+' |','|---|'+'---:|'*len(METHODS)]
 for c,rs in bycase.items():lines.append('| '+c+' | '+' | '.join(f"{next(r for r in rs if r['method']==m)['lpips']:.5f}" for m in METHODS)+' |')
 protocol=json.loads((OUT/'protocol.json').read_text())
 lines+=['','## 冻结配置与实现','','所有交集均为7层×3head×1step=21/1392单元，预算1.51%；按层7/29=24.14%，按head和step各25%。层0不参与，noisy step从0编号。','']
 for m,p in protocol['policies'].items():lines.append(f"- {LABELS[m]}：layers={p.get('layers','all')}，heads={p.get('heads','all')}，steps={p.get('steps','all')}。")
 lines+=['','- [搜索方法、选择及正式计时](../intersection_search_v1/REPORT.zh.md)','- [冻结清单及策略文件SHA](protocol.json)','- [完整20组逐样本数据](paired_results.csv)、[汇总](summary.csv)、[逐场景](by_prompt.csv)','- [搜索执行代码](../../../local_experiments/video_runtime_v1/intersection_search.py)','- [验证执行代码](../../../local_experiments/video_runtime_v1/intersection_validation.py)','- [本报告生成代码](../../../local_experiments/video_runtime_v1/report_intersection_validation.py)','', '视频：每组目录下保存7份video.mp4、latents.pt及result.json；comparison.mp4为两行对比，上排Dense/原交集/搜索均值/搜索均衡，下排按层/按head/按step。','']
 for c in CASES:lines.append('- '+c+'：'+'、'.join(f'[seed{s}](videos/{c}_s{s}/comparison.mp4)' for s in SEEDS))
 (OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,5))
 for i,m in enumerate(METHODS[1:]):
  values=[r['lpips_delta_vs_original'] for r in rows if r['method']==m];ax.scatter([i+(j%4-1.5)*.035 for j in range(20)],values,alpha=.6,s=20);ax.plot(i,statistics.mean(values),'kd',ms=7)
 ax.axhline(0,color='gray');ax.set_xticks(range(5),METHODS[1:],rotation=15);ax.set_ylabel('LPIPS difference vs original intersection (lower is better)');fig.tight_layout();fig.savefig(OUT/'paired_lpips.png',dpi=160);plt.close(fig)
 def movie(pair):
  c,s=pair;folder=OUT/'videos'/f'{c}_s{s}';target=folder/'comparison.mp4'
  if target.exists():return
  names=['dense']+METHODS;cmd=['ffmpeg','-v','error','-y','-filter_complex_threads','1']
  for m in names:cmd+=['-i',str(folder/m/'video.mp4')]
  filters=[f"[{i}:v]scale=384:256,drawtext=text='{m}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[v{i}]" for i,m in enumerate(names)]
  filters+=[''.join(f'[v{i}]' for i in range(7))+'xstack=inputs=7:layout=0_0|384_0|768_0|1152_0|0_256|384_256|768_256:fill=black[out]']
  subprocess.run(cmd+['-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-threads','2','-crf','20',str(target)],check=True)
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(movie,[(c,s) for c in CASES for s in SEEDS]))
 print(json.dumps(table,indent=2))
if __name__=='__main__':main()
