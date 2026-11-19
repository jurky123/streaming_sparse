"""Finalize the fourth kernel optimization from completed benchmark evidence."""
import json,statistics,hashlib,shutil,datetime
from pathlib import Path
source=Path(__file__).resolve().parents[1];out=source/'artifacts/comparison_20261113';raw=Path('/tmp/latest_kernel_optimization4');archive=out/'integration_checks/kernel_optimization4'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
assert read(out/'audit.json')['passed'] and read(out/'audit.json')['completed']==36
archive.mkdir(parents=True,exist_ok=True)
for p in raw.glob('*.json'):shutil.copy2(p,archive/p.name)
for name in ['final_smoke','candidates']:shutil.copytree(raw/name,archive/name,dirs_exist_ok=True)
(archive/'INPUTS.txt').write_text('Microbenchmarks reuse /tmp/latest_kernel_optimization2/summary_inputs.pt; same real model snapshots as round 2. Full benchmark uses each scene/shared_inputs.pt. Timings are diagnostics on spare GPUs; use full runs for end-to-end conclusions.\n')
smoke=read(archive/'final_smoke/smoke.json');assert smoke['status']=='passed';checks=smoke['storage']['summary_validation'];assert len(checks)==45
maxerr=max(e['nmse'] for r in checks for e in r['errors'].values())
for f in ['three_term_summary_checks.json','summary_stress_checks.json']:assert read(archive/f)['passed']
rows=read(out/'summary.json')['rows'];index={(r['case'],r['method']):r for r in rows};cases=['horse_field','traffic_crossing']
def avg(m,key='total_seconds'):return statistics.mean(index[c,m][key] for c in cases)
lines=['最新算法第四轮优化：融合summary与代表组分块','',f'主路径含VAE平均耗时：{avg("ours_latest_rope"):.3f} → {avg("ours_latest_summary"):.3f} 秒，耗时减少 {(1-avg("ours_latest_summary")/avg("ours_latest_rope"))*100:.2f}%。',f'更快tail路径：{avg("ours_latest_rope_tail"):.3f} → {avg("ours_latest_summary_tail"):.3f} 秒，耗时减少 {(1-avg("ours_latest_summary_tail")/avg("ours_latest_rope_tail"))*100:.2f}%。','', '相同最新summary_pretrained/no_downstream算法与权重；512×768、3 latent帧/chunk、15帧历史/360页、41 chunks、480 RGB帧。相同输入、seed101、0.1密度及16 probes。','每场景完整预热后两轮；生成和原生VAE均计时；模型加载、编译、视频编码及质量指标不计入。主路径GPU0/1，tail路径GPU3/5；分别与同卡Dense配对。','', '实现：','1. 将summary拆成页特征、16-token代表组pooling、页均值/Haar detail、代表组风险统计四个Triton kernel，减少PyTorch中间张量及寄存器溢出。','2. BF16来源的K/V直接参与tensor-core投影；FP32投影权重分解为三项BF16，FP32累加。K投影后中心化在实数下与原式等价，但浮点顺序变化。','3. 完整128维均值、3×8 Haar细节、四风险公式、最新参数、原始分组及选页策略保持；输出仍为BF16。','4. 原64-query精确attention与两级流水保持；16/32-query实验收益不足，未接入。','',f'完整41 chunks验证通过；45组真实summary抽检（含第20/40 chunk），最大字段NMSE={maxerr:.9g}。常量、近常量、参数变化测试通过。','微基准cheap约1.39→0.28 ms，full约1.84→0.36 ms；局部倍数不能直接作为端到端加速比。','浮点归约顺序变化会改变后续选页和生成轨迹。本版本不逐位等价；上一轮ours_latest_rope仍保留用于要求旧输出的场景。','']
for method,previous in [('ours_latest_summary','ours_latest_rope'),('ours_latest_summary_tail','ours_latest_rope_tail')]:
 drift={};deltas={}
 lines.append(method)
 for case in cases:
  r=index[case,method];old=index[case,previous];data=read(out/case/method/'result.json');q=read(out/case/method/'quality_vs_reference.json');assert q['reference']==previous;drift[case]=q
  lines.append(f"{case}: 生成{r['generation_seconds']:.3f}s，含VAE {r['total_seconds']:.3f}s，峰值{r['peak_gib']:.3f}GiB；同卡Dense加速{r['speedup']:.3f}x。")
  deltas[case]={}
  for w in ['early','middle','late']:
   delta={k:r[w+'_'+k]-old[w+'_'+k] for k in ['lpips','ssim','psnr']};deltas[case][w]=delta
   lines.append(f"  {w} vs Dense: LPIPS {r[w+'_lpips']:.6f} (变化{delta['lpips']:+.6f}), SSIM {r[w+'_ssim']:.6f}, PSNR {r[w+'_psnr']:.3f}; 直接vs上一版 LPIPS {q['windows'][w]['lpips']:.6f}")
  lines.append(f"  完整latent逐位相同={q['bitwise_equal']}，latent NMSE={q['latent_nmse']:.6g}；两轮计时="+str([round(x['total_generation_vae_seconds'],4) for x in data['repetitions']]))
 acceptance=dict(status='numerical_gates_passed_full_video_drift_measured',algorithm='summary_pretrained/no_downstream',algorithm_and_checkpoint_match=True,history_pages=360,full_chunks=41,summary_checks=len(checks),summary_max_nmse=maxerr,reference=previous,drift=drift,quality_vs_dense_delta=deltas,bitwise_equivalent=all(q['bitwise_equal'] for q in drift.values()),mean_total_seconds=avg(method),mean_previous_seconds=avg(previous),paired_dense_geometric_speedup=statistics.geometric_mean(index[c,method]['speedup'] for c in cases),interpretation='Numerical candidate; measured video drift is not a lossless replacement or an absolute quality guarantee.')
 write(out/(method.replace('ours_','')+'_acceptance.json'),acceptance)
lines+=['','质量指标仅表示接近同输入Dense或上一版本，不等于绝对视觉质量。两个场景/一个seed是小样本；长视频后段Dense和稀疏版本本身都有退化。结合contact图和完整视频判断。','画面对照（第61/240/440帧）：主路径中后段仍有明显背景斑块与纹理异常；horse_field的新tail中后段相较旧tail出现更严重的斑点。因此不能根据相对Dense的LPIPS降低宣称绝对画质改善，新tail保留为速度候选。','优化后真实profile：noisy前向约166.7ms GPU kernel时间，主要热点转为主干GEMM约44.9ms、tail约29.7ms、精确attention约22.1ms。clean提交约280.8ms，其中FlashAttention约113.2ms。该profile含采样开销，不替代完整计时。','下一轮优先空间：tail与精确attention、clean提交的attention/分组。VAE约20s为所有方法共同开销；若优化VAE，需同步更新对照方法。']
(out/'KERNEL_OPTIMIZATION4.zh.txt').write_text('\n'.join(lines)+'\n')
manifest=read(out/'manifest.json');manifest.update(completed=36,expected=36,audit_passed=True,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
paths=list((source/'local_experiments/review_v1').glob('*.py'))+[source/'local_experiments'/n for n in ['stream_eval_latest_summary.py','stream_eval_latest_summary_tail.py']]+list((source/'local_setup').glob('*summary*.py'))
hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for key,impl in [('latest_summary','SummaryLatestRuntime'),('latest_summary_tail','SummaryTailLatestRuntime')]:manifest[key]=dict(implementation=impl,code_sha256=hashes,acceptance=key+'_acceptance.json',report='KERNEL_OPTIMIZATION4.zh.txt',checks='integration_checks/kernel_optimization4',bitwise_equivalent=False,role='faster numerical candidate; previous same-output path retained')
write(out/'manifest.json',manifest)
status=read(out/'status.json');status.update(status='complete',completed=36,expected=36,audit_passed=True,latest_summary='complete; numerical gates passed; video drift measured',latest_summary_tail='complete; numerical gates passed; video drift measured');write(out/'status.json',status)
print('\n'.join(lines[:5]))
