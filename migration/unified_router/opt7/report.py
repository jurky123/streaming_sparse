"""Consolidate measured timings, eight native-frame validations and profiles."""
import argparse,json,statistics,hashlib,gzip,collections,subprocess
from pathlib import Path
import numpy as np

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();out=root/'optimization7';repo=Path(__file__).resolve().parents[3]
 load=lambda f:json.loads(Path(f).read_text())
 manifest=load(root/'e2e_manifest.json');pids=manifest['prompt_ids'];paired=load(out/'paired/summary.json');alltimings=[]
 for i,pid in enumerate(pids):
  folder=out/'paired'if i==0 else out/f'paired_val/prompt{i}'
  assert load(folder/'status.json')['status']=='completed'
  times=load(folder/'summary.json');records=load(folder/'timings.json')['records']
  dense=times['Dense']['generation_seconds'];fast=times['Opt7-Reuse0']['generation_seconds']
  alltimings.append(dict(prompt_id=pid,dense_seconds=dense,fast_seconds=fast,dense_total_seconds=times['Dense']['total_seconds'],fast_total_seconds=times['Opt7-Reuse0']['total_seconds'],speedup=dense/fast,latency_reduction=1-fast/dense))
 variants={'Opt7-Full':'GuardCachedSplitCounters','Opt7-Score':'ScoreCompactCachedSplitCounters','Opt7-Reuse0':'ScoreCompactCachedReuse0SplitCounters'}
 quality={};baseline=[];per_prompt=[]
 for pid in pids:
  result=load(out/f'quality/{pid}.seed0/result.json');baseline.append(result['baseline_vs_dense']['sparse_frames_only']['lpips'])
  prompt=dict(prompt_id=pid,baseline_lpips=baseline[-1],variants={})
  for label,variant in variants.items():
   row=result['variants'][variant];assert row['first_chunk_identical']
   lpips=row['quality_vs_dense']['sparse_frames_only']['lpips'];delta=lpips-baseline[-1]
   r=dict(lpips=lpips,delta_lpips=delta,keep=row['weighted_keep'],lpips_to_opt6=row['quality_vs_opt6']['sparse_frames_only']['lpips'],psnr=row['quality_vs_dense']['sparse_frames_only']['psnr_db'],av_nmse=max(x['masked_attention_nmse']for x in row['attention_audit']))
   quality.setdefault(label,[]).append(r);prompt['variants'][label]=r
  per_prompt.append(prompt)
 summary={label:dict(lpips_mean=statistics.mean(r['lpips']for r in rows),lpips_delta_mean=statistics.mean(r['delta_lpips']for r in rows),lpips_delta_worst=max(r['delta_lpips']for r in rows),keep_mean=statistics.mean(r['keep']for r in rows),max_masked_attention_nmse=max(r['av_nmse']for r in rows),psnr_mean=statistics.mean(r['psnr']for r in rows))for label,rows in quality.items()}
 profiles={}
 for name in ['GuardCachedSplitCounters_step3','ScoreCompactCachedReuse0SplitCounters_step0','ScoreCompactCachedReuse0SplitCounters_step3']:
  row=load(out/f'profile_{name}/kernel_attribution.json');profiles[name]=dict(total_ms=row['total_kernel_ms'],modules={r['module']:r['kernel_ms']for r in row['modules']})
 before=load(root/'optimization6/guard_profile/kernel_attribution.json');profiles['Opt6_step3']=dict(total_ms=before['total_kernel_ms'],modules={r['module']:r['kernel_ms']for r in before['modules']})
 sync={}
 for label,folder in [('Opt6',root/'optimization6/guard_profile'),('Opt7-Full',out/'profile_GuardCachedSplitCounters_step3')]:
  events=json.load(gzip.open(folder/'trace.json.gz'))['traceEvents'];sync[label]=dict(collections.Counter(e['name']for e in events if e.get('cat')in ['cuda_runtime','cuda_driver']and any(t in e['name'].lower()for t in ['synchronize','memcpy'])))
 mean_dense=statistics.mean(r['dense_seconds']for r in alltimings);mean_fast=statistics.mean(r['fast_seconds']for r in alltimings)
 timing_summary=dict(dense_mean_seconds=mean_dense,fast_mean_seconds=mean_fast,speedup=mean_dense/mean_fast,latency_reduction=1-mean_fast/mean_dense,wins=sum(r['speedup']>1 for r in alltimings),prompts=len(pids))
 result=dict(scope='8 Val prompts, seed 0, native 45 RGB frames; quality over 36 sparsified frames; Test unused',paired_first_prompt=paired,all8_timing=timing_summary,per_prompt_timing=alltimings,baseline_lpips_mean=statistics.mean(baseline),quality=summary,per_prompt_quality=per_prompt,profiles=profiles,synchronization=sync,counters=load(out/'counters_probe.json'))
 write(out/'final_summary.json',result)
 # Validate frozen baseline/training artifacts; sources only added in opt7.
 guard=load(root/'optimization6/guard_manifest.json');frozen={**manifest['frozen_files'],**manifest['code_sha256'],**guard['code'],**guard['prior_unchanged']}
 for path,expected in frozen.items():assert digest(path)==expected,path
 sources={str(f):digest(f)for f in sorted((repo/'migration/unified_router/opt7').glob('*.py'))};sources[str(repo/'v10_optimization7_progress.sh')]=digest(repo/'v10_optimization7_progress.sh')
 video_checks={}
 for name in ['Full','Score','Reuse0']:
  path=out/f'ALL8_Dense_Opt6_Opt7_{name}.mp4'
  video= json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)]))['streams'][0]
  assert int(video['nb_read_frames'])==360 and video['width']==2496 and video['height']==480,video
  video_checks[name]=video
 write(out/'final_verification.json',dict(frozen_files_verified=len(frozen),all24_quality_outputs_complete=True,first_chunk_identical_all24=True,masked_attention_audits=216,audit_queries=216*144,max_masked_attention_nmse=max(r['max_masked_attention_nmse']for r in summary.values()),videos=video_checks,test_used=False,training_performed=False,accepted_opt6_default_unchanged=True))
 write(out/'manifest.json',dict(code=sources,prior_unchanged=frozen,baseline='Opt6 Guarded K32',variants=variants,default_changed=False,query_stride=1,first_chunk_dense=True,layer0_dense=True,current_prototypes=16,history_prototypes=[8,4,2],global_tf32=False,quality_scope='Val only; per-query masks; original KV final attention',fast_variant_approximations=['AV action correction removed','route refreshed at denoise step 0 only; step 1/2/3 reuse, within each chunk only']))
 # Export standalone figures.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=list(paired);fig,axes=plt.subplots(1,3,figsize=(16,4.5))
 axes[0].bar(names,[paired[n]['generation_seconds']for n in names],color=['#555555','#c45c4c','#da9261','#66a7b8','#397984']);axes[0].tick_params(axis='x',rotation=25);axes[0].set_ylabel('Seconds');axes[0].set_title('Generation: same GPU, prompt 1, median of 6')
 qnames=['Opt6',*variants];axes[1].bar(qnames,[statistics.mean(baseline)]+[summary[n]['lpips_mean']for n in variants],color=['#c45c4c','#da9261','#66a7b8','#397984']);axes[1].tick_params(axis='x',rotation=25);axes[1].set_ylabel('LPIPS to Dense (lower is closer)');axes[1].set_title('8 Val prompts: 36 sparsified frames')
 x=np.arange(8);axes[2].bar(x-.18,[r['dense_seconds']for r in alltimings],.36,label='Dense',color='#555555');axes[2].bar(x+.18,[r['fast_seconds']for r in alltimings],.36,label='Opt7-Reuse0',color='#397984');axes[2].set_xticks(x,[str(i+1)for i in x]);axes[2].set_xlabel('Val prompt');axes[2].set_ylabel('Generation seconds');axes[2].set_title('Each prompt measured on the same GPU');axes[2].legend()
 fig.tight_layout();fig.savefig(out/'speed_quality.png',dpi=180);fig.savefig(out/'speed_quality.pdf');plt.close(fig)
 # Human-readable report, generated from raw measurements.
 lines=['# Opt7：KV 循环拆分、统计融合与路由简化实验','',f'结论：保留完整路由时，首个固定样例生成耗时由 {paired["Opt6"]["generation_seconds"]:.3f}s 降到 {paired["Opt7-Full"]["generation_seconds"]:.3f}s。采用精简路由并仅在 step0 刷新后，8 个 Val 样例平均 {mean_fast:.3f}s，Dense 平均 {mean_dense:.3f}s，耗时降低 {(1-mean_fast/mean_dense)*100:.2f}%。这仍接近 Dense，距离 15% 逻辑保留率所暗示的 attention 计算收益有明显差距。最快版改变了路由算法，不能称为完整 Opt6 的等价提速。','', '## 同卡端到端计时','', '首个样例：每种方法单独预热，6 轮正反交替顺序；包含全部 4 个 chunk、4 个去噪步和 context 重建。生成不含 VAE；总耗时包含 VAE 与像素回传。首 chunk 与 layer0 保持 Dense。','', '| 方法 | 生成/s | 含 VAE/s | 路由策略 |','|---|---:|---:|---|']
 policies={'Dense':'全量 attention','Opt6':'已接受的 Guarded K32 完整路由','Opt7-Full':'完整路由 + Split-KV + 历史投影缓存 + 统计融合','Opt7-Score':'去掉 AV 修正头；每步独立页评分','Opt7-Reuse0':'精简页评分；每 chunk 只在 step0 刷新'}
 for n in names:lines.append(f'| {n} | {paired[n]["generation_seconds"]:.3f} | {paired[n]["total_seconds"]:.3f} | {policies[n]} |')
 lines+=['','8 个样例的 Dense/最快版均在各自同一张 A100 上交替测量；第 1 个 6 轮，其余各 3 轮。以下取每个样例的中位数：','','| 样例 | Dense/s | Reuse0/s | Dense / Reuse0 |','|---|---:|---:|---:|']
 for i,r in enumerate(alltimings):lines.append(f'| {i+1}: {r["prompt_id"]} | {r["dense_seconds"]:.3f} | {r["fast_seconds"]:.3f} | {r["speedup"]:.3f}× |')
 lines+=['',f'最快版在 {timing_summary["wins"]}/8 个样例耗时低于 Dense；均值比值 {timing_summary["speedup"]:.3f}×。不能从第一个样例的小幅优势推断所有样例均显著更快。','', '## 代码改动与热点证据','', '1. **拆分长 KV 循环。** 每组 32 个 query 保留各自 mask，把页并集分成两段并行执行在线 softmax，再合并部分输出与 log-sum-exp。代表块用 4 warps / 2 stages；attention 从 Opt6 的约 2.943 ms 降至 2.567 ms。每个 query 的可见页没有共享或广播，不是此前被否定的 Q64 共享路由。浮点累加顺序发生改变。','2. **融合保留率统计。** 直接归约 bool mask 与页成本，避免写出 66,493,440 bytes 的浮点乘积；两段 kernel 直接输出计数和比例。真实 mask 下计数与比例完全一致，单次主机墙钟从约 0.415 ms 降至 0.088 ms。旧 trace 每层 3 次 cudaStreamSynchronize，新版 0 次；单独这项带来约 0.10–0.12s 整段收益，4 组对照 latent 逐位一致。','3. **缓存历史投影。** 用仍存活的历史 Part 对象身份验证缓存，当前段每步重算。代表层投影从约 0.298 ms 降到 0.112 ms。不同 prompt/历史对象不会复用旧投影。','4. **精简评分是算法消融。** 去掉候选选择、内容能量、集合均值、81 动作 AV 修正头；只计算独立 query 对每页的 logsumexp 得分和 mean + beta×std 阈值。历史 P2/P4/P8 不再填充到 P16；不物化 prototype logits、weights 或候选内容张量。beta 仍为 1.09。','5. **跨步复用也是算法近似。** 每个 chunk 的 step0 用当前 Q/K 生成独立 mask，step1–3 只复用 mask；实际 Q、原始 K/V、softmax 和 AV 每步都重算。set_mode 清空本次生成的 mask 缓存。没有跨 prompt 使用答案，也没有保留对首 chunk 的特殊可见性。','', '## 视频质量与正确性','', '相同 8 个 Val prompt、seed0、45 帧原生 480×832 / 16fps；统计使用后 36 帧，避免相同的首 9 帧稀释误差。LPIPS 衡量与 Dense 的距离，不等于视觉好坏或时序稳定性的完整评价。','','| 方法 | LPIPS 均值↓ | 相对 Opt6 均值变化 | 最坏单例变化 | 保留率 |','|---|---:|---:|---:|---:|',f'| Opt6 | {statistics.mean(baseline):.6f} | 0 | 0 | 约 15.16% |']
 for name,r in summary.items():lines.append(f'| {name} | {r["lpips_mean"]:.6f} | {r["lpips_delta_mean"]:+.6f} | {r["lpips_delta_worst"]:+.6f} | {r["keep_mean"]*100:.3f}% |')
 lines+=['',f'24 次生成的首 chunk latent 与首 9 帧像素均与 Dense 逐位一致。216 个实际轨迹块、31,104 个 query/head 检查与同 mask 的 SDPA 输出，最大 NMSE 为 {max(r["max_masked_attention_nmse"]for r in summary.values()):.3g}。这证明 attention kernel 数值正确；它不证明跨步路由复用没有质量损失。', '', '最快版的平均 Dense 距离没有恶化，但 Johannesburg 机器人等个例偏差增大，SUV 与海豹的后续轨迹/姿态也与 Opt6 有可见差异。评审帧仅覆盖末帧，不据此宣称整段时序质量相同。完整视频在下方。没有重新训练，也没有使用 Test 选择方案。','', '## 剩余瓶颈','', '| Layer24 / chunk4 | Opt6 step3 | Opt7-Full step3 | Reuse0 step0 | Reuse0 step3 |','|---|---:|---:|---:|---:|']
 ps=[profiles['Opt6_step3'],profiles['GuardCachedSplitCounters_step3'],profiles['ScoreCompactCachedReuse0SplitCounters_step0'],profiles['ScoreCompactCachedReuse0SplitCounters_step3']]
 for module in ['route_graph_copy_and_replay','sparse_attention_including_union','all_prototypes','current_prototypes','prototype_projection','other_backbone_and_diagnostics','ffn','cross_attention']:
  lines.append('| '+module+' | '+' | '.join(f'{r["modules"].get(module,0):.3f}'for r in ps)+' |')
 lines.append('| GPU kernel 合计/ms | '+' | '.join(f'{r["total_ms"]:.3f}'for r in ps)+' |')
 lines+=['','GPU 归因按 CUDA runtime/driver correlation 回溯到所属模块，避免 CUDA graph 在 CPU scope 下的漏计；profile 与计时分开运行。此处是代表层，不能直接乘层数冒充端到端。', '', '当前 15% 是每个 query 的逻辑保留率。真实代表块中，32-query 页并集约 44.37%；Tensor Core 实际执行的是这个并集，不是 15%。即使取消全部路由，旧 attention 的同轨迹 oracle 仍需约 4.457s，Dense 约 4.519s：原 attention 根本没有留下足够空间支付路由成本。Oracle 预先保存同一轨迹的 mask，只用于诊断上限，不能部署或计入实用加速。', '', '新完整路由仍占约 5.48ms，而 attention 相对 Dense 的节省只有数百微秒；只继续微调 81 动作头难以获得数量级收益。下一步应同时约束实际页并集成本和真实 AV 误差：训练更便宜的页评分/预算修正器，并评估能降低并集的候选池或分组约束，仍允许每个 query 独立选页。对跨步复用，应重点验证中后步的刷新规则。上述是后续方向，本轮没有把未训练方案当成已验证结果。', '', '## 排除的方案','', '多页合并与显式循环流水、多种 group/warps/stages/Split 配置、二维/时空 query 重排均已实测。部分多页循环配置在当前 Triton 编译器崩溃，其余通常更慢；二维重排未降低并集。只采用实测最优的原始 query 顺序 + Split2。失败与负结果保留在 attention_sweep*、native_sweep、spatial_probes、tune*.json。','', '在线 softmax/分块实现参考 [Triton fused-attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) 与 [循环流水接口](https://triton-lang.org/main/python-api/generated/triton.language.range.html)；本轮 kernel 为独立实现。','', '## 查看与复现','', '- [全部 8 段：Dense / Opt6 / 完整路由优化](ALL8_Dense_Opt6_Opt7_Full.mp4)','- [全部 8 段：Dense / Opt6 / 精简评分](ALL8_Dense_Opt6_Opt7_Score.mp4)','- [全部 8 段：Dense / Opt6 / 最快复用方案](ALL8_Dense_Opt6_Opt7_Reuse0.mp4)','- [速度与质量图](speed_quality.png)、[PDF](speed_quality.pdf)、[需重点检查的末帧](reuse_review_frames.jpg)','- [原始汇总](final_summary.json)、[校验](final_verification.json)、[代码及冻结文件 SHA](manifest.json)','', '运行状态：仓库根目录 `./v10_optimization7_progress.sh --once`。','', '核心代码：`migration/unified_router/opt7/split_attention.py`、`counters.py`、`fused_network.py`、`projection_cache.py`、`compact_score.py`、`reuse.py`。新入口为 `opt7.variant`，通过 `SLIMAV_VARIANT` 显式选择实验；原 Opt6 入口没有被替换。','', '复现同卡测试：','```bash','CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2 \\','  streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap \\','  migration.unified_router.opt7.paired_bench \\',f'  --directory {root} --repeats 6','```','']
 (out/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps(dict(paired=paired,all8=timing_summary,quality=summary,verification_files=len(frozen)),indent=2))
if __name__=='__main__':main()
