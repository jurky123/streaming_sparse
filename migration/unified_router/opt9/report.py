"""Measured Opt9 results, sealed dependencies, quality and kernel attribution."""
import argparse,json,hashlib,statistics,collections,subprocess
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();out=root/'optimization9';repo=Path(__file__).resolve().parents[3];pids=read(root/'e2e_manifest.json')['prompt_ids']
 names=['Dense','Opt8Full','Packed','Skip2','Skip0','Opt8Fast','Skip2Fast','Skip0Fast'];new=['Packed','Skip2','Skip0','Skip2Fast','Skip0Fast','MemoryFull','MemoryFast'];timings=[]
 for i,pid in enumerate(pids):
  folder=out/('paired'if i==0 else f'paired_val/prompt{i}');assert read(folder/'status.json')['status']=='completed';timings.append(dict(prompt_id=pid,summary=read(folder/'summary.json')))
 timing={}
 for name in names:
  metrics={key:statistics.mean(t['summary'][name][key]for t in timings)for key in ['generation_seconds','decode_seconds','total_seconds']}
  metrics.update(dense_speedup=statistics.mean(t['summary']['Dense']['generation_seconds']for t in timings)/metrics['generation_seconds'],wins_vs_dense=sum(t['summary'][name]['generation_seconds']<t['summary']['Dense']['generation_seconds']for t in timings));timing[name]=metrics
 quality={}
 for name in ['Opt8Full','Opt8Fast']:
  old='QKProj'if name=='Opt8Full'else'QKFast';rows=[read(root/f'optimization8/quality/{pid}.seed0/{old}.json')for pid in pids]
  quality[name]=dict(lpips_mean=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['lpips']for r in rows),keep_mean=statistics.mean(r['weighted_keep']for r in rows),max_attention_nmse=max(x['masked_attention_nmse']for r in rows for x in r['attention_audit']))
 for name in new:
  rows=[read(out/f'quality/{pid}.seed0/{name}.json')for pid in pids];audits=[x for r in rows for x in r['attention_audit']];baseline='Opt8Fast'if name.endswith('Fast')else'Opt8Full'
  old='QKFast'if name.endswith('Fast')else'QKProj';oldlp=[read(root/f'optimization8/quality/{pid}.seed0/{old}.json')['quality_vs_dense']['sparse_frames_only']['lpips']for pid in pids]
  assert all(r['first_chunk_identical']for r in rows)
  checked=sum(x.get('route_checked_queries',0)for x in audits);changed=sum(x.get('route_changed_queries',0)for x in audits)
  quality[name]=dict(lpips_mean=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['lpips']for r in rows),lpips_delta_mean=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['lpips']-v for r,v in zip(rows,oldlp)),lpips_delta_worst=max(r['quality_vs_dense']['sparse_frames_only']['lpips']-v for r,v in zip(rows,oldlp)),baseline=baseline,keep_mean=statistics.mean(r['weighted_keep']for r in rows),max_attention_nmse=max(x['masked_attention_nmse']for x in audits),route_changed_queries=changed,route_checked_queries=checked,attention_audited_queries=sum(x['queries']for x in audits),route_changed_query_fraction=changed/checked if checked else None)
 memory_timings=[]
 for i,pid in enumerate(pids):
  folder=out/('memory_paired'if i==0 else f'memory_paired_val/prompt{i}');assert read(folder/'status.json')['status']=='completed';memory_timings.append(dict(prompt_id=pid,summary=read(folder/'summary.json')))
 memory_timing={}
 primary_names=['Dense','Opt8Full','Packed','MemoryFull','Opt8Fast','MemoryFast']
 for name in primary_names:
  metrics={key:statistics.mean(t['summary'][name][key]for t in memory_timings)for key in ['generation_seconds','decode_seconds','total_seconds']}
  metrics.update(dense_speedup=statistics.mean(t['summary']['Dense']['generation_seconds']for t in memory_timings)/metrics['generation_seconds'],wins_vs_dense=sum(t['summary'][name]['generation_seconds']<t['summary']['Dense']['generation_seconds']for t in memory_timings));memory_timing[name]=metrics
 for name in ['MemoryFull','MemoryFast']:
  assert all(read(out/f'quality/{pid}.seed0/{name}.json')['kv_rewrite_bit_identical']for pid in pids)
 profiles={}
 for f in sorted(out.glob('profile_*/kernel_attribution.json')):
  data=read(f);groups=collections.defaultdict(float)
  for r in data['kernels']:groups[r['kernel']]+=r['milliseconds']
  result=dict(total_ms=data['total_kernel_ms'],modules={r['module']:r['kernel_ms']for r in data['modules']},kernels=dict(sorted(groups.items(),key=lambda x:-x[1])))
  if(f.parent/'skip_stats.json').exists():result['skip_stats']=read(f.parent/'skip_stats.json')
  profiles[f.parent.name.removeprefix('profile_')]=result
 for n in ['Opt8Full_step3','Packed_step3','Skip2_step3','Skip0_step3','Skip2Fast_step3','Skip2Fast_step0','Skip0Fast_step3','MemoryFull_step3','MemoryFast_step3','MemoryFast_step0']:assert n in profiles,n
 summary=dict(scope='8 Val prompts seed0, 4 native chunks x 4 steps, Test unused',all8_timing=timing,paired_first_prompt=timings[0]['summary'],per_prompt_timing=timings,quality=quality,profiles=profiles,memory_timing=memory_timing,memory_per_prompt_timing=memory_timings,recommended_full='MemoryFull',recommended_fast='MemoryFast',training_performed=False)
 write(out/'final_summary.json',summary)
 old=read(root/'optimization8/manifest.json');frozen={**old['code'],**old['prior_unchanged']}
 for path,digest in frozen.items():assert sha(path)==digest,path
 initial=read(out/'implementation_lock.json')['code'];core=['select.py','network.py','skip_attention.py']
 for n in core:
  path=str(repo/'migration/unified_router/opt9'/n);assert sha(path)==initial[path],path
 for path,digest in read(out/'memory_implementation_lock.json').items():assert sha(path)==digest,path
 codes={str(f):sha(f)for f in sorted((repo/'migration/unified_router/opt9').glob('*.py'))};codes[str(repo/'v10_optimization9_progress.sh')]=sha(repo/'v10_optimization9_progress.sh')
 write(out/'manifest.json',dict(code=codes,prior_unchanged=frozen,memory_core_lock=read(out/'memory_implementation_lock.json'),runtime=read(out/'runtime_options.json'),core_lock={str(repo/'migration/unified_router/opt9'/n):initial[str(repo/'migration/unified_router/opt9'/n)]for n in core},default_existing_entries_changed=False))
 videos={}
 for name in new:
  f=out/f'ALL8_Dense_Opt8Full_Opt9_{name}.mp4';d=read_json_process(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(f)])['streams'][0]
  assert int(d['nb_read_frames'])==360 and d['width']==2496 and d['height']==480;videos[name]=d
 assert read(out/'checksum_verification.json')['all_match_quality_including_dense_and_opt8']
 write(out/'final_verification.json',dict(timed_outputs_verified=read(out/'checksum_verification.json')['timed_outputs_checked'],frozen_files_verified=len(frozen),new_source_files=len(codes),quality_outputs=56,memory_rewrite_bit_identical_all16=True,first_chunk_identical_all=True,timed_new_outputs_equal_quality_outputs=True,core_unchanged_during_quality_and_timing=True,videos=videos,test_used=False,training_performed=False))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,3,figsize=(17,4.7));colors=['#555555','#bd6d56','#d1a061','#3b8790','#779fc1','#90a88a','#4c7957','#92b260']
 axes[0].bar(primary_names,[memory_timing[n]['generation_seconds']for n in primary_names],color=colors[:6]);axes[0].tick_params(axis='x',rotation=50);axes[0].set_ylabel('Generation seconds');axes[0].set_title('8 Val: mean of per-prompt medians')
 modules=['route_graph_copy_and_replay','sparse_attention_including_union','other_backbone_and_diagnostics','ffn','cross_attention'];x=np.arange(5)
 for shift,name,color in [(-.2,'Opt8Full_step3','#bd6d56'),(.2,'MemoryFull_step3','#3b8790')]:axes[1].bar(x+shift,[profiles[name]['modules'].get(k,0)for k in modules],.4,label=name,color=color)
 axes[1].set_xticks(x,['Router','Sparse attn','Other / copies','FFN','Cross attn'],rotation=25);axes[1].set_ylabel('GPU kernel ms');axes[1].set_title('L24 / chunk4 / step3');axes[1].legend()
 for name,color in zip(primary_names[1:],colors[1:]):axes[2].scatter(memory_timing[name]['generation_seconds'],quality[name]['lpips_mean'],label=name,color=color)
 axes[2].set_xlabel('Generation seconds');axes[2].set_ylabel('LPIPS distance to Dense (36 sparse frames)');axes[2].set_title('Measured speed / error');axes[2].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'speed_hotspots.png',dpi=170);fig.savefig(out/'speed_hotspots.pdf');plt.close(fig)
 full=memory_timing['MemoryFull'];fast=memory_timing['MemoryFast'];dense=memory_timing['Dense'];oldfull=memory_timing['Opt8Full'];oldfast=memory_timing['Opt8Fast']
 lines=['# Opt9：候选归约、KV 量化与 V 整理融合','',f'完整路由平均生成时间 {oldfull["generation_seconds"]:.3f} → {full["generation_seconds"]:.3f}s，降低 {(1-full["generation_seconds"]/oldfull["generation_seconds"])*100:.2f}%。Fast 分支 {oldfast["generation_seconds"]:.3f} → {fast["generation_seconds"]:.3f}s；Dense 为 {dense["generation_seconds"]:.3f}s，Fast 相对 Dense 加速 {fast["dense_speedup"]:.3f}×（延迟降低 {(1-fast["generation_seconds"]/dense["generation_seconds"])*100:.2f}%）。','', '最终采用 `MemoryFull` / `MemoryFast`：去掉 BF16 K 拼接，并将 V 整理融合进 K 量化；内存改写相对 Packed / Opt8Fast 的 16 次生成全部 latent 逐位一致。在线 PV 跳过作为消融保留：-2 在整网没有稳定额外收益，0 的质量代价明显，均不作为最终默认。完整路由版仍慢于 Dense，尚未达到理论稀疏计算量对应的性能。','', '## 在线跳过等消融的端到端速度','', '8 个 Val prompt，seed0；每个 prompt 的全部 8 种模式在同一张 A100 上交替执行，先各自预热，再正反顺序轮换。第 1 个 prompt 6 次，其余各 3 次；表中为逐 prompt 中位数的均值。generation 包含 4 chunks × 4 denoise steps、原样历史上下文重建和 latent 回传；total 加上 VAE 和像素回传。不含首次编译。各新模式的计时输出与独立质量评估逐位一致。','','| 模式 | 生成/s | 含 VAE/s | 相对 Dense 速度比 | 含义 |','|---|---:|---:|---:|---|']
 desc={'Dense':'BF16 全量基线','Opt8Full':'上一版 QKProj，完整 AV 路由，每步刷新','Packed':'只优化候选选择','Skip2':'Packed + 在线跳过，log2 阈值 -2','Skip0':'Packed + 更激进阈值 0','Opt8Fast':'上一版 QKFast，无 AV 头，step0 mask 复用','Skip2Fast':'Opt8Fast + 阈值 -2','Skip0Fast':'Opt8Fast + 阈值 0'}
 for n in names:
  t=timing[n];lines.append(f'| {n} | {t["generation_seconds"]:.3f} | {t["total_seconds"]:.3f} | {t["dense_speedup"]:.3f}× | {desc[n]} |')
 lines+=['',f'MemoryFast 在最终阶段的 {fast["wins_vs_dense"]}/8 个样例快于 Dense。Fast 没有 AV 修正头且复用去噪步路由；它的速度不能当作完整学习路由的速度。','', '## 本轮实现和未采用方案','', '1. **32 位候选 key。** 将非负 rank 的 FP32 bit pattern 与页号打包成一个 32 位整数，一次整数最小值归约完成排序和稳定选择；保留原候选配额、顺序与缺位回填。为页号舍弃少量低精度 rank 位，因此是近似候选选择。代表 512-query tile 从约 0.107ms 降至 0.066ms；6,144 个 query 的初始 mask 不变，1 行添加候选发生变化。精确 64 位打包和通用 argmin 更慢；精确两次整数归约有部分收益但不及 packed32。','2. **在线 PV 跳过。** 借鉴 [SpargeAttention](https://arxiv.org/html/2502.18137v4) 和 [BLASST](https://arxiv.org/html/2512.12087v1)：QK 后用块最大值相对 running maximum 判断是否跳过 exp、V 读取和 PV。独立 query mask 保持，只有整个 CTA 的可见 query 都满足条件才跳过；分子分母同时不更新。代表性 attention -2 阈值约 2.15→1.87ms，0 阈值约 1.63ms；实际端到端和分层误差如下。','3. **历史 prototype 去补零。** 实现真实 P2/P4/P8/P16 QK、片上补齐 PV 的 6 种配置，没有稳定优于原内核，未接入最终路径。','4. **保守阈值与库适配。** -12/-8 等较保守阈值节省的 PV 不足以抵消分支开销；现有公开 block-sparse API 的块 mask 不能直接保持当前逐 query 语义，未把公开库宣传速度作为本项目收益。[调研与设计详情](RESEARCH_NOTES.md)。','', '## 8 段视频与实际误差','', '下表 LPIPS 仅统计后 36 个稀疏生效帧，表示与 Dense 的距离。首 chunk latent 和前 9 帧像素在 56 次新生成中均与 Dense 一致；每个方案评估 360 原生帧。量化/在线跳过误差按真实 QKV、同一个路由 mask 的 BF16 SDPA 输出抽检（layer1/24/29、3 个稀疏 chunk、step3，各 144 个 query），不是代理 score loss。','','| 模式 | LPIPS↓ | 相对同分支 Opt8 | 逻辑保留率 | 最大抽检 attention NMSE |','|---|---:|---:|---:|---:|']
 for n in names[1:]+['MemoryFull','MemoryFast']:
  r=quality[n];lines.append(f'| {n} | {r["lpips_mean"]:.6f} | {r.get("lpips_delta_mean",0):+.6f} | {r["keep_mean"]*100:.3f}% | {r["max_attention_nmse"]*100:.3f}% |')
 lines+=['', '阈值 0 的局部最坏误差和视频偏离明显增大，作为激进备选保留；最终选择没有在线跳过的 Memory 路径。未重训，不保证训练一定能恢复误差。LPIPS 的小幅下降也不等于普遍视觉质量更好。候选选择的同 QKV 路由差异率、逐 prompt / 逐帧指标见 `final_summary.json` 和 `quality/`。','', '## 优化后热点','', '独立 profiled pass：layer24、第 4 chunk、step3；GPU kernel 归属来自 trace 的 CPU scope / correlation 映射，不能将 CPU scope 总时长误当 GPU kernel 耗时。','','| 模块 | Opt8Full/ms | Packed/ms | Skip2/ms | Skip2Fast step3/ms |','|---|---:|---:|---:|---:|']
 for key in ['route_graph_copy_and_replay','sparse_attention_including_union','ffn','cross_attention','prototype_projection','current_prototypes']:
  values=[profiles[n]['modules'].get(key,0)for n in ['Opt8Full_step3','Packed_step3','Skip2_step3','Skip2Fast_step3']];lines.append('| '+key+' | '+' | '.join(f'{v:.3f}'for v in values)+' |')
 lines+=['','Skip2 主要 kernel（完整列表在 profile 目录）：','','| kernel | ms |','|---|---:|']
 for name,value in list(profiles['Skip2_step3']['kernels'].items())[:12]:lines.append(f'| `{name}` | {value:.3f} |')
 lines+=['','实际页面工作量（单独 warmup 插桩，没有放入计时或 profiler pass）：','','| 模式 | QK 页 tile 密度 | PV 页 tile 密度 | 相对已路由 QK 跳过的 PV tile |','|---|---:|---:|---:|']
 for n in ['Skip2_step3','Skip0_step3','Skip2Fast_step3','Skip2Fast_step0','Skip0Fast_step3']:
  r=profiles[n]['skip_stats'];lines.append(f'| {n} | {r["qk_tile_density"]*100:.2f}% | {r["pv_tile_density"]*100:.2f}% | {r["pv_tiles_skipped_fraction"]*100:.2f}% |')
 lines+=['', '页 tile 密度以 G32×KV64 调度为单位，相对于全部页；包含边界 padding，不是精确 token FLOP 比例。逻辑约 15% 的 mask 在 CTA 页并集后会放大；在线跳过减少 PV，但仍需为该并集计算 QK。','', '下一处优先看代理 `_score_energy` 和集合统计：候选选择缩短后，完整路由仍有大量评分、FP16 prototype 权重写读及均值计算。Fast 路由复用后则主要剩 attention、FFN、cross attention 和 KV/RoPE/逐元素操作。后续可探索不改变逐 query 决策的空间分组、页式 KV 与历史量化缓存；公共 backbone 优化必须同步用于 Dense 对照。','', '## 结果与复现','', '[最终完整路由视频：Dense / Opt8Full / MemoryFull](ALL8_Dense_Opt8Full_Opt9_MemoryFull.mp4) · [最终 Fast 视频](ALL8_Dense_Opt8Full_Opt9_MemoryFast.mp4) · [激进完整路由视频](ALL8_Dense_Opt8Full_Opt9_Skip0.mp4) · [激进 Fast 视频](ALL8_Dense_Opt8Full_Opt9_Skip0Fast.mp4)','', '视频中列固定为 **Dense、Opt8Full、标注的 Opt9 变体**；Fast 误差表按其自身 Opt8Fast 基线比较。','', '[速度/热点图](speed_hotspots.png) · [PDF](speed_hotspots.pdf) · [完整数字](final_summary.json) · [校验结果](final_verification.json) · [源码封存](manifest.json) · [训练对齐要求](TRAINING_ALIGNMENT.md)','', '新增代码：`migration/unified_router/opt9/`；旧代码、checkpoint 和数据未改。查看：`./v10_optimization9_progress.sh --once`。','', '```bash','SLIMAV_OPT9=MemoryFull CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\','  TORCHINDUCTOR_COMPILE_THREADS=2 streaming_sparse/.venv/bin/python -u \\','  -m migration.runtime.bootstrap migration.unified_router.opt9.memory_network \\',f'  --directory {root} \\',f'  --output {out}/reproduce_MemoryFull --mode K32-Set','```','', 'Fast：改为 `SLIMAV_OPT9=MemoryFast` 和 `--mode U-Sol`。仅新增入口，不替换既有默认实验。','']
 memory_lines=['## 最终采用方案：独立同卡复测','', '最终阶段重新对 Dense、Opt8 和 Memory 路径进行同卡交替测速；以下均为 8 Val 的逐 prompt 中位数均值，重复次数仍为 6/3。与后面的第一阶段消融分开列出，避免混用不同阶段的基线。','','| 模式 | 生成/s | 含 VAE/s | 相对 Dense |','|---|---:|---:|---:|']
 for name in primary_names:
  t=memory_timing[name];memory_lines.append(f'| {name} | {t["generation_seconds"]:.3f} | {t["total_seconds"]:.3f} | {t["dense_speedup"]:.3f}× |')
 memory_lines+=['','MemoryFull = Packed + 融合 K 量化 / V 整理；MemoryFast = Opt8Fast + 相同访存优化。最终方案没有新增在线 softmax 剪枝。','','代表性片段（包含原 KV 拼接）的三种布局：直接读 K/V 为约 2.39ms，只整理 V 为约 2.34ms，融合 K 量化与 V 整理为约 2.23ms；原版约 2.52ms。三个方案的微基准输出均逐位一致。V 的连续布局仍有价值，因此没有机械地消灭所有中间张量，而是去掉不必要的 BF16 K 和多余 launch/读写。','','最终热点对照：','','| 模块 | Opt8Full/ms | MemoryFull/ms | MemoryFast step3/ms |','|---|---:|---:|---:|']
 for key in ['route_graph_copy_and_replay','sparse_attention_including_union','other_backbone_and_diagnostics','ffn','cross_attention']:
  values=[profiles[n]['modules'].get(key,0)for n in ['Opt8Full_step3','MemoryFull_step3','MemoryFast_step3']];memory_lines.append('| '+key+' | '+' | '.join(f'{v:.3f}'for v in values)+' |')
 memory_lines+=['', '注意：原 KV cat 归入 other，新版 V 整理归入 attention，必须结合两项看，不能仅比较 attention 这一项。最终完整块 GPU kernel 合计 '+f'{profiles["Opt8Full_step3"]["total_ms"]:.3f} → {profiles["MemoryFull_step3"]["total_ms"]:.3f}ms。','', 'MemoryFull 完整 block 的前几项 kernel 热点：','']
 for name,value in list(profiles['MemoryFull_step3']['kernels'].items())[:8]:memory_lines.append(f'- `{name}`：{value:.3f}ms')
 memory_lines+=['']
 lines[6:6]=memory_lines
 (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(timing=memory_timing,quality=quality),indent=2))
def read_json_process(cmd):return json.loads(subprocess.check_output(cmd))
if __name__=='__main__':main()
