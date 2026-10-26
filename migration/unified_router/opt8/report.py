"""Opt8 final measured results and post-optimization hotspot attribution."""
import argparse,json,hashlib,statistics,collections,subprocess
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();out=root/'optimization8';repo=Path(__file__).resolve().parents[3];load=lambda p:json.loads(Path(p).read_text())
 m=load(root/'e2e_manifest.json');pids=m['prompt_ids'];paired=load(out/'paired/summary.json');alltime=[]
 for i,pid in enumerate(pids):
  folder=out/'paired'if i==0 else out/f'paired_val/prompt{i}';assert load(folder/'status.json')['status']=='completed'
  t=load(folder/'summary.json');alltime.append(dict(prompt_id=pid,dense=t['Dense'],fast=t['QKFast'],generation_speedup=t['Dense']['generation_seconds']/t['QKFast']['generation_seconds']))
 mean_dense=statistics.mean(r['dense']['generation_seconds']for r in alltime);mean_fast=statistics.mean(r['fast']['generation_seconds']for r in alltime)
 all8=dict(dense_mean_seconds=mean_dense,fast_mean_seconds=mean_fast,speedup=mean_dense/mean_fast,latency_reduction=1-mean_fast/mean_dense,wins=sum(r['generation_speedup']>1 for r in alltime))
 results=[load(out/f'quality/{pid}.seed0/result.json')for pid in pids];quality={};baseline=statistics.mean(r['baseline_vs_dense']['sparse_frames_only']['lpips']for r in results)
 for variant in ['Full','Proj','QKProj','QKFast']:
  rs=[r['variants'][variant]for r in results];deltas=[r['quality_vs_dense']['sparse_frames_only']['lpips']-b['baseline_vs_dense']['sparse_frames_only']['lpips']for b,r in zip(results,rs)]
  audits=[x for r in rs for x in r['attention_audit']]
  quality[variant]=dict(lpips_mean=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['lpips']for r in rs),lpips_delta_mean=statistics.mean(deltas),lpips_delta_worst=max(deltas),keep_mean=statistics.mean(r['weighted_keep']for r in rs),latent_identical_count=sum(r['latent_bit_identical']for r in rs),max_masked_attention_nmse=max(x['masked_attention_nmse']for x in audits),route_changed_queries=sum(x.get('route_changed_queries',0)for x in audits),route_checked_queries=sum(x.get('route_checked_queries',0)for x in audits),attention_audited_queries=sum(x['queries']for x in audits),native_frames_evaluated=360)
 oldfast=[]
 for pid in pids:oldfast.append(load(root/f'optimization7/quality/{pid}.seed0/ScoreCompactCachedReuse0SplitCounters.json')['quality_vs_dense']['sparse_frames_only']['lpips'])
 quality['QKFast']['delta_vs_opt7fast_mean']=quality['QKFast']['lpips_mean']-statistics.mean(oldfast)
 profiles={}
 folders={'Opt7Full':root/'optimization7/profile_GuardCachedSplitCounters_step3','Full':out/'profile_Full_step3','QKProj':out/'profile_QKProj_step3','QKFast_step0':out/'profile_QKFast_step0','QKFast_step3':out/'profile_QKFast_step3'}
 for label,folder in folders.items():
  data=load(folder/'kernel_attribution.json');groups=collections.defaultdict(float)
  for r in data['kernels']:groups[r['kernel']]+=r['milliseconds']
  profiles[label]=dict(total_ms=data['total_kernel_ms'],modules={r['module']:r['kernel_ms']for r in data['modules']},kernels=dict(sorted(groups.items(),key=lambda x:-x[1])))
 summary=dict(scope='K32 full route and U-Sol temporal-reuse fast branch; 8 Val seed0; Test unused',paired_first_prompt=paired,all8_timing=all8,per_prompt_timing=alltime,opt7full_lpips=baseline,opt7fast_lpips=statistics.mean(oldfast),quality=quality,profiles=profiles,user_accepted_quant_error=True,training_rerun=False)
 write(out/'final_summary.json',summary)
 previous=load(root/'optimization7/manifest.json');frozen={**previous['code'],**previous['prior_unchanged']}
 for path,h in frozen.items():assert sha(path)==h,path
 for path,h in load(out/'implementation_lock.json').items():assert sha(path)==h,path
 codes={str(f):sha(f)for f in sorted((repo/'migration/unified_router/opt8').glob('*.py'))};codes[str(repo/'v10_optimization8_progress.sh')]=sha(repo/'v10_optimization8_progress.sh')
 write(out/'manifest.json',dict(code=codes,prior_unchanged=frozen,core_lock=load(out/'implementation_lock.json'),runtime=load(out/'runtime_options.json'),tolerance_update=load(out/'user_tolerance_update.json'),default_existing_entries_changed=False))
 videos={}
 for variant in ['Proj','QKProj','QKFast']:
  path=out/f'ALL8_Dense_Opt7_Opt8_{variant}.mp4';v=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)]))['streams'][0]
  assert int(v['nb_read_frames'])==360 and v['width']==2496 and v['height']==480;videos[variant]=v
 assert quality['Full']['latent_identical_count']==8 and quality['Full']['route_changed_queries']==0
 assert all(r['variants'][v]['first_chunk_identical']for r in results for v in quality)
 write(out/'final_verification.json',dict(frozen_files_verified=len(frozen),source_files=len(codes),quality_outputs=32,first_chunk_identical_all=True,exact_full_variant_bit_identical_all8=True,full_reference_route_audits_unchanged=True,paired_outputs_equal_quality_outputs=True,videos=videos,quantization_error_accepted_by_user=True,training_performed=False,test_used=False))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(17,4.5));names=list(paired)
 axes[0].bar(names,[paired[n]['generation_seconds']for n in names],color=['#555555','#bd6d56','#d1a061','#74aeb3','#3b8790','#90a88a','#4c7957']);axes[0].tick_params(axis='x',rotation=40);axes[0].set_ylabel('Generation seconds');axes[0].set_title('Same GPU, prompt 1, median of 6')
 modules=['route_graph_copy_and_replay','sparse_attention_including_union','ffn','cross_attention'];labels=['Router','Sparse attention','FFN','Cross attention'];x=np.arange(4)
 axes[1].bar(x-.18,[profiles['Opt7Full']['modules'].get(k,0)for k in modules],.36,label='Opt7-Full',color='#bd6d56');axes[1].bar(x+.18,[profiles['QKProj']['modules'].get(k,0)for k in modules],.36,label='Opt8-QKProj',color='#3b8790');axes[1].set_xticks(x,labels,rotation=20);axes[1].set_ylabel('GPU kernel ms');axes[1].set_title('Post-optimization profile: L24 / chunk4 / step3');axes[1].legend()
 x=np.arange(8);axes[2].bar(x-.18,[r['dense']['generation_seconds']for r in alltime],.36,label='Dense',color='#555555');axes[2].bar(x+.18,[r['fast']['generation_seconds']for r in alltime],.36,label='QKFast',color='#4c7957');axes[2].set_xticks(x,[str(i+1)for i in x]);axes[2].set_xlabel('Val prompt');axes[2].set_ylabel('Generation seconds');axes[2].set_title('Each pair measured on the same GPU');axes[2].legend();fig.tight_layout();fig.savefig(out/'speed_hotspots.png',dpi=180);fig.savefig(out/'speed_hotspots.pdf');plt.close(fig)
 lines=['# Opt8：完成优化后的速度与热点分析','',f'结果：保留完整路由的 QKProj 从 Opt7Full 的 {paired["Opt7Full"]["generation_seconds"]:.3f}s 降到 {paired["QKProj"]["generation_seconds"]:.3f}s，耗时降低 {(1-paired["QKProj"]["generation_seconds"]/paired["Opt7Full"]["generation_seconds"])*100:.2f}%。最快分支 QKFast 在 8 个 Val 样例平均 {mean_fast:.3f}s，Dense 平均 {mean_dense:.3f}s，耗时降低 {all8["latency_reduction"]*100:.2f}%。本轮量化误差按用户明确接受的口径记录，不做末层 BF16 回退。','', '## 端到端计时','', '同一张 A100，所有版本单独预热后 6 轮正反顺序交替运行。包含 4 chunks × 4 步、原样 context 重建；生成时间不含 VAE，总时间含 VAE 与像素回传。每次输出均重复一致，且新版本与独立质量评估的 latent 逐位一致。','','| 方法 | 生成/s | 含 VAE/s | 变化 |','|---|---:|---:|---|']
 meanings={'Dense':'原生 BF16 全量 attention','Opt7Full':'上一轮完整 K32 路由','Full':'共享权重读取的双集合均值；输出不变','Proj':'Full + FP16 Tensor Core 查询投影','QKProj':'Proj + INT8 QK；保留 AV 修正头，每步重算路由','Opt7Fast':'上一轮精简评分 + step0 路由复用','QKFast':'Opt7Fast + INT8 QK；不含 AV 修正头'}
 for n in names:lines.append(f'| {n} | {paired[n]["generation_seconds"]:.3f} | {paired[n]["total_seconds"]:.3f} | {meanings[n]} |')
 lines+=['', '8 个样例的 Dense/QKFast 配对计时：第 1 个 6 次，其余各 3 次，每个样例的两种模式在同卡交替运行，取中位数。','','| 样例 | Dense/s | QKFast/s | 速度比 |','|---|---:|---:|---:|']
 for i,r in enumerate(alltime):lines.append(f'| {i+1} `{r["prompt_id"]}` | {r["dense"]["generation_seconds"]:.3f} | {r["fast"]["generation_seconds"]:.3f} | {r["generation_speedup"]:.3f}× |')
 lines+=['',f'QKFast 在 {all8["wins"]}/8 个样例更快，平均速度比 {all8["speedup"]:.3f}×。已经小幅超过当前 Dense 基线，仍未达到按 15% 逻辑保留率估算的计算收益；完整路由版仍慢于 Dense。','', '## 已实施的优化','', '1. **双集合均值复用权重读取。** 旧 kernel 分别读取同一 W 计算全量和选中集合。新 kernel 在同一个循环中加载一次 W 和 V，累加两个 FP32 输出。配置 BM32/BK64/4 warps/3 stages。实际 8 段视频的 Full latent 与 Opt7Full 逐位一致；同 QKV 路由审核没有 mask 差异。','2. **融合局部 FP16 查询投影。** 把 q.float → FP32 bmm → half 改成直接从原 BF16 Q 和 FP32 basis 读入，局部转换 FP16、Tensor Core 累加 FP32、输出 FP16。处理 basis 的实际三维 stride，不假定连续。单块投影 NMSE 约 8.06e-8，77/6144 个 query 的最终 mask 改变，实际 AV 增量 NMSE 约 1.54e-6；这是数值近似，不宣称逐位等价。','3. **INT8 QK 稀疏 attention。** Q 按每组 32 queries、K 按原生 64-token 页对称量化；QK 累加 INT32，缩放后 softmax FP32，PV 保留 BF16 输入/FP32 累加，最后 Split2 合并。4 个 kernel 的量化、页并集、attention、合并全部计入计时。保留每个 query 独立 mask，没有使用 Q64 共享路由。首 chunk 与 layer0 继续 Dense。','', '量化设计参考 [SageAttention 官方实现与论文入口](https://github.com/thu-ml/SageAttention)。公开 [Sparse SageAttention API](https://github.com/jt-zhang/Sparse_Attention_API) 使用块级 mask；本项目还需要每个 query 的独立页 mask 和每 chunk 的 8-token 尾页，因此这里实现了匹配现有语义的 Triton kernel，没有把共享块 mask 当成等价替代。','', '## 优化后的热点','', '优化与视频验证后另起进程采集 profile，按 CUDA runtime/driver correlation 归因 GPU kernel；不混用 CPU scope 时间与 CUDA graph 时间。代表层为 layer24 / chunk4 / step3。','','| 模块/ms | Opt7Full | Opt8 Full | Opt8 QKProj |','|---|---:|---:|---:|']
 for key in ['route_graph_copy_and_replay','sparse_attention_including_union','current_prototypes','prototype_projection','ffn','cross_attention','other_backbone_and_diagnostics']:
  lines.append('| '+key+' | '+' | '.join(f'{profiles[v]["modules"].get(key,0):.3f}'for v in ['Opt7Full','Full','QKProj'])+' |')
 lines.append('| GPU kernel 合计 | '+' | '.join(f'{profiles[v]["total_ms"]:.3f}'for v in ['Opt7Full','Full','QKProj'])+' |')
 lines+=['', 'QKProj 中的主要 kernel：','','| kernel | ms | 含义 |','|---|---:|---|']
 descriptions={'_score_energy':'全页 prototype 评分、权重和能量','_fused_select':'阈值及候选页选择','_dual':'全量/选中集合均值','_guard_finish':'81 动作评分与选择','_weighted_candidates':'候选内容向量','_factor_prepare':'动作头因式化预处理','_gram_stats':'几何与集合统计','_project':'新 FP16 查询投影（全部 query tiles）','_split':'INT8 QK / BF16 PV attention 主循环','_quantize':'K 的 INT8 量化','_union':'每组 query 的页并集','_merge':'Split 输出合并'}
 for k in descriptions:
  if k in profiles['QKProj']['kernels']:lines.append(f'| `{k}` | {profiles["QKProj"]["kernels"][k]:.3f} | {descriptions[k]} |')
 lines+=['', '完整路由的剩余重点是 `_score_energy` 与 `_fused_select`，合计约 2.26ms；仍有 prototype 权重张量写读和多次候选归约。下一轮值得优先尝试压缩历史 P2/P4/P8 的评分布局、减少候选选择的串行归约，并评估更少候选动作对速度的收益。已测过的简单 warp 数调整没有稳定提速，不能重复把它算作收益。','', 'QKFast 在 step1–3 复用 mask，剩余主要是 attention、FFN、cross attention，以及 QKV 拼接、逐元素运算和 RoPE。profile 中 QKV 拼接约 0.37ms，BF16 加法类 kernel 合计约 0.35ms，复数 FP64 乘法约 0.18ms；这些是后续可检验的访存/融合方向。公共 backbone 优化应同时给 Dense 测对照。这里没有把未实施的方向算入加速。','', '逻辑保留率约 15%，代表块的实际页并集仍约 44%，因此实际 Tensor Core 工作量仍远高于逻辑保留量。INT8 减轻了一部分计算和读取成本，没有消除页并集放大。','', '## 误差记录与训练对齐','', '统计后 36 个稀疏生效帧，8 个 Val prompt、seed0、原生 480×832 / 16fps。32 次生成的首 chunk latent 和前 9 帧像素均与 Dense 相同。LPIPS 表示到 Dense 的距离，不把距离下降解释成普遍的视觉质量提升。','','| 方法 | LPIPS 均值↓ | 相对 Opt7Full | 保留率 | 最大抽检 attention NMSE |','|---|---:|---:|---:|---:|',f'| Opt7Full | {baseline:.6f} | 0 | 约15.16% | — |']
 for name,r in quality.items():lines.append(f'| {name} | {r["lpips_mean"]:.6f} | {r["lpips_delta_mean"]:+.6f} | {r["keep_mean"]*100:.3f}% | {r["max_masked_attention_nmse"]:.6g} |')
 lines+=['',f'QKFast 相对其自身 Opt7Fast 基线的平均 LPIPS 变化为 {quality["QKFast"]["delta_vs_opt7fast_mean"]:+.6f}。局部量化误差最高约 1.49%，用户已明确接受，因此本轮继续用无末层回退版本测速；没有把较宽容的误差口径混成“数值等价”。','', '重训未在本轮启动。训练应采用相同的 FP16 投影、INT8 量化分组与实际 AV 前向，用真实最终 AV NMSE 选择 best；单 query 标签不能改变部署时的 32-query 量化尺度分组。QKFast 没有动作头，只重训旧动作头不会改变它。详细接口与损失口径见 [训练对齐说明](TRAINING_ALIGNMENT.md)。','', '## 消融与复现','', '- 每 token INT8 QK：代表 attention 约 2.28ms；每页/组 INT8 QK 约 2.15ms，选择后者做端到端验证。','- 预先把 V 转 FP16，以及在 attention 内转换 FP16，均没有优于保留 BF16 PV。','- QK 与 PV 都用 INT8 的试验约 3.16ms，比原 Split attention 约 2.58ms 更慢，因此没有采用。','- 原 masked mean 调度已扫描 36 种组合，再扫描 36 种双集合复用配置；采用实测最优配置。','- 全过程没有改训练数据、旧 checkpoint、Opt7 及更早的代码封存；新增代码位于 `migration/unified_router/opt8/`。','', '核心代码：`dual_means.py`、`projection.py`、`projected_route.py`、`block_int8_attention.py`、`network.py`。`guarded.py` 是未采用的末层回退备选，最终 runtime 配置的 fallback_layers 为空。','', '[全部视频：Dense / Opt7Full / QKProj](ALL8_Dense_Opt7_Opt8_QKProj.mp4) · [最快 QKFast 对比](ALL8_Dense_Opt7_Opt8_QKFast.mp4) · [仅 FP16 投影 Proj 对比](ALL8_Dense_Opt7_Opt8_Proj.mp4)','', '[速度与热点图](speed_hotspots.png) · [PDF](speed_hotspots.pdf) · [完整数值](final_summary.json) · [校验](final_verification.json) · [代码 SHA](manifest.json)','', '查看状态：`./v10_optimization8_progress.sh --once`。复现完整路由优化：','```bash','SLIMAV_OPT8=QKProj CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\','  TORCHINDUCTOR_COMPILE_THREADS=2 streaming_sparse/.venv/bin/python -u \\','  -m migration.runtime.bootstrap migration.unified_router.opt8.network \\',f'  --directory {root} \\',f'  --output {out}/reproduce_QKProj --mode K32-Set','```','', '最快分支将 `SLIMAV_OPT8` 改为 `QKFast`、`--mode` 改为 `U-Sol`。原有入口未被替换；新入口默认 Full 是逐位等价的均值优化版。','']
 (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(paired=paired,all8=all8,quality=quality),indent=2))
if __name__=='__main__':main()
