"""Build a report only from completed benchmark artifacts."""
import json,csv,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/comparison_20261113'
METHODS=['dense','ours_latest_summary','ours_latest_summary_tail','ours_latest_rope','ours_latest_rope_tail','ours_latest_cached','ours_latest_cached_tail','ours_latest_stable','ours_latest_optimized','ours_latest_fused','ours_latest_fast','ours','dasa','pisa','solattn','svgear','lightforcing','forcingkv'];CASES=['horse_field','traffic_crossing'];rows=[];missing=[]
for case in CASES:
 dense=OUT/case/'dense/result.json'
 reference=json.loads(dense.read_text()) if dense.exists() else None
 for method in METHODS:
  path=OUT/case/method/'result.json'
  if not path.exists():missing.append(f'{case}/{method}');continue
  data=json.loads(path.read_text());reps=data['repetitions'];assert len(reps)==2
  reference=json.loads((OUT/data['paired_dense']).read_text()) if data.get('paired_dense') else json.loads(dense.read_text()) if dense.exists() else None
  times=[r['total_generation_vae_seconds'] for r in reps];gen=[r['generation_seconds'] for r in reps]
  if reference:
   assert all(r['metadata']['inputs_sha256']==reference['repetitions'][0]['metadata']['inputs_sha256'] for r in reps)
   assert all(r['metadata']['checkpoint_sha256']==reference['repetitions'][0]['metadata']['checkpoint_sha256'] for r in reps)
   assert all(r['physical_gpu']==reference['repetitions'][0]['physical_gpu'] for r in reps)
  quality=data['quality'];st=reps[0].get('storage') or {}
  row=dict(case=case,method=method,generation_seconds=statistics.mean(gen),total_seconds=statistics.mean(times),total_min=min(times),total_max=max(times),speedup=statistics.mean(r['total_generation_vae_seconds'] for r in reference['repetitions'])/statistics.mean(times) if reference else None,peak_gib=max(r['peak_allocated_bytes'] for r in reps)/2**30,kv_gib=st.get('self_kv_storage_bytes',st.get('raw_bytes',0))/2**30)
  row['persistent_cache_and_aux_gib']=st.get('persistent_self_kv_and_aux_bytes',st.get('persistent_bytes',st.get('self_kv_storage_bytes',0)+st.get('auxiliary_bytes',0)))/2**30
  row['physical_gpu']=reps[0]['physical_gpu']
  row['paired_dense_seconds']=statistics.mean(r['total_generation_vae_seconds'] for r in reference['repetitions']) if reference else None
  row['auxiliary_gib']=st.get('auxiliary_bytes',max(0,st.get('persistent_bytes',0)-st.get('raw_bytes',0)))/2**30
  for window in ['early','middle','late']:
   q=quality.get('windows',{}).get(window,quality if window=='early' else {})
   for metric in ['lpips','ssim','psnr']:row[f'{window}_{metric}']=q.get(metric)
  rows.append(row)
result=dict(complete=not missing,missing=missing,rows=rows)
OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.json').write_text(json.dumps(result,indent=2))
if rows:
 with (OUT/'summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['统一配置视频推理比较','',f'状态：{"全部完成" if not missing else "尚未完成"}', '512×768；每chunk 3 latent帧；共同历史容量上限15 latent帧（压缩方法保留策略见说明）；41 chunks；480输出帧；16fps。','相同Self-Forcing generator_ema、文本embedding及初始/转移噪声；seed101。','8张同型号A100并行；每个方法与同一张物理卡上的Dense配对计算加速比。新增设备的Dense输出必须与原参考逐位相同。完整预热后两轮，生成与VAE分别计时，持续检查GPU进程。','', '说明：','1. ours为当前summary_pretrained，历史容量从338页调整为360页以统一窗口；摘要额外占用单独保留，非等内存比较。','2. ours为最新算法FP32参考路径；ours_latest_fast使用同一summary_pretrained权重、360历史页、摘要和路由公式，复用page64核与输入搬运并适配完整均值/零和细节补偿。路由Graph保持相同FP32计算；attention浮点顺序改变并已做数值验证。GainVector64是不同算法，不参与此项复测。','3. 主缓存总容量统一18帧（15历史+3当前）；Forcing-KV按官方head策略保留更少历史；DASA压缩历史并额外保留3帧clean KV及特征缓存。','4. PISA/Sol-Attn使用官方等长内核加Query填充并裁剪；填充成本包含在测量中，非优化矩形实现。','5. SVG-EAR保留官方聚类/误差路由/融合代码，修改KV长度视图并使用固定FlashInfer。','6. 质量指标衡量相对同输入dense的接近程度，不等于绝对视觉质量或文本一致性。','7. 两场景、一个seed、两轮耗时是小样本实测，不代表论文榜单或统计显著性。','8. 同时间点画面检查显示Dense长视频后段自身也会退化；ours中后段存在明显斑点/纹理异常。相对Dense指标不能替代绝对视觉质量评价，结合contact.png与完整视频判断。','10. ours_latest_fast与ours不逐位相同；完整视频存在明显数值累积漂移，不能作为无损替换。直接差异见各场景/ours_latest_fast/quality_vs_reference.json和latest_fast_acceptance.json。','11. ours_latest_fused在相同最新算法与attention实现上，将风险路由融合为3个Triton kernel；浮点归约顺序有变化。真实模型抽检选页一致，完整视频漂移和性能见KERNEL_OPTIMIZATION.zh.txt。','12. ours_latest_optimized继续使用相同算法，合批72个tile的路由、对原始摘要计算使用执行图，并对tail使用双BF16项与FP32累加。前两项逐位验证通过；attention及完整视频变化见KERNEL_OPTIMIZATION2.zh.txt。','13. ours_latest_stable只使用摘要执行图和路由合批，保留ours_latest_fused的attention。完整输出等价检查见各场景/ours_latest_stable/equivalence_vs_fused.json。','14. ours_latest_cached复用原KV缓存的当前帧空闲区，并对原始分组和选页使用执行图；精确attention采用两级流水。cached_tail在更快tail候选上应用相同优化。结果与完整输出等价记录见KERNEL_OPTIMIZATION3.zh.txt。','15. ours_latest_rope及rope_tail进一步融合Q/K RoPE，保留FP64计算和原转换路径；与各自上一版本的完整视频等价检查和性能见KERNEL_OPTIMIZATION3.zh.txt。','16. ours_latest_summary及summary_tail使用4个Triton kernel融合原summary公式，三项BF16权重投影与FP32累加改变浮点顺序；长序列数值验证、完整视频漂移与性能见KERNEL_OPTIMIZATION4.zh.txt。','9. PSNR使用原分辨率，LPIPS/SSIM在256×384计算；各窗口80帧。质量主表使用第0轮视频，SVG-EAR另存重复性及第1轮指标。','', '场景 方法 生成秒 含解码秒 加速倍数 前段LPIPS 前段SSIM 前段PSNR']
for r in rows:lines.append(f"{r['case']} {r['method']} {r['generation_seconds']:.3f} {r['total_seconds']:.3f} {r['speedup']:.3f} {r['early_lpips']} {r['early_ssim']} {r['early_psnr']}")
if missing:lines+=['','未完成：',*missing]
lines+=['','两场景平均（耗时包含VAE；加速比为同卡配对比值的几何均值）：','方法 平均秒 加速比 前段LPIPS 后段LPIPS']
aggregate=[]
for method in METHODS:
 rr=[r for r in rows if r['method']==method]
 if len(rr)!=2:continue
 item=dict(method=method,total_seconds=statistics.mean(r['total_seconds'] for r in rr),speedup_geomean=statistics.geometric_mean(r['speedup'] for r in rr),early_lpips=statistics.mean(r['early_lpips'] for r in rr) if method!='dense' else 0,late_lpips=statistics.mean(r['late_lpips'] for r in rr) if method!='dense' else 0,peak_gib=max(r['peak_gib'] for r in rr))
 aggregate.append(item)
 lines.append(f"{method} {item['total_seconds']:.2f} {item['speedup_geomean']:.3f} {item['early_lpips']:.4f} {item['late_lpips']:.4f}")
lines+=['','抽帧视觉核查见VISUAL_REVIEW.zh.txt及两个contact.png；完整视频位于各场景/方法/video.mp4。']
(OUT/'aggregate.json').write_text(json.dumps(dict(scope='Mean across two scenes; speedup geometric mean uses same-device paired Dense; quality repetition zero',rows=aggregate),indent=2))
(OUT/'REPORT.zh.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(completed=len(rows),total=len(CASES)*len(METHODS),missing=missing)))
