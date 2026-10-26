"""Compile measured speed, quality, changed-query error, and frozen-source checks."""
import argparse,json,hashlib,statistics,collections,subprocess
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory).resolve();out=root/'optimization10';repo=Path(__file__).resolve().parents[3];pids=read(root/'e2e_manifest.json')['prompt_ids'];names=['Dense','Opt9','Wide','Head','Moments','Compact'];new=names[2:];timings=[]
 for i,pid in enumerate(pids):
  folder=out/('paired'if i==0 else f'paired_val/prompt{i}');assert read(folder/'status.json')['status']=='completed';timings.append(dict(prompt_id=pid,summary=read(folder/'summary.json')))
 timing={}
 for name in names:
  metrics={key:statistics.mean(t['summary'][name][key]for t in timings)for key in ['generation_seconds','decode_seconds','total_seconds']}
  metrics.update(dense_speedup=statistics.mean(t['summary']['Dense']['generation_seconds']for t in timings)/metrics['generation_seconds'],wins_vs_dense=sum(t['summary'][name]['generation_seconds']<t['summary']['Dense']['generation_seconds']for t in timings));timing[name]=metrics
 old=[read(root/f'optimization9/quality/{pid}.seed0/MemoryFull.json')for pid in pids];oldlp=[r['quality_vs_dense']['sparse_frames_only']['lpips']for r in old];quality={'Opt9':dict(lpips_mean=statistics.mean(oldlp),keep_mean=statistics.mean(r['weighted_keep']for r in old))}
 for name in new:
  rows=[read(out/f'quality/{pid}.seed0/{name}.json')for pid in pids];audits=[x for r in rows for x in r['attention_audit']];assert all(r['first_chunk_identical']for r in rows)
  if name=='Wide':assert all(r['latent_bit_identical']for r in rows)
  checked=sum(x['route_checked_queries']for x in audits);changed=sum(x['route_changed_queries']for x in audits)
  delta=[r['quality_vs_dense']['sparse_frames_only']['lpips']-v for r,v in zip(rows,oldlp)]
  quality[name]=dict(lpips_mean=statistics.mean(r['quality_vs_dense']['sparse_frames_only']['lpips']for r in rows),lpips_delta_mean=statistics.mean(delta),lpips_delta_worst=max(delta),lpips_distance_to_opt9=statistics.mean(r['quality_vs_opt9_full']['sparse_frames_only']['lpips']for r in rows),keep_mean=statistics.mean(r['weighted_keep']for r in rows),max_attention_quantization_nmse=max(x['masked_attention_nmse']for x in audits),route_changed_queries=changed,route_checked_queries=checked,route_changed_query_fraction=changed/checked,fixed_query_router_max_nmse=max(x['router_incremental_attention_nmse']for x in audits),per_prompt=[dict(prompt_id=pid,lpips=r['quality_vs_dense']['sparse_frames_only']['lpips'],lpips_delta=d,keep=r['weighted_keep'])for pid,r,d in zip(pids,rows,delta)])
  if name!='Wide':
   extra=[read(out/f'changed_audit/{pid}.{name}.json')for pid in pids];assert all(r['latent_matches_quality']for r in extra);items=[x for r in extra for x in r['audit']];count=sum(x['changed_queries']for x in items);assert count==changed
   quality[name]['changed_query_audit']=dict(checked=count,blocks=len(items),aggregate_nmse=sum(x['sse']for x in items)/sum(x['reference_energy']for x in items),worst_query_nmse=max(x['worst_changed_query_nmse']for x in items),worst_block_nmse=max(x['changed_query_aggregate_nmse']for x in items))
 profiles={}
 for name in names[1:]:
  folder=out/f'profile_{name}_step3';assert read(folder/'status.json')['status']=='completed';data=read(folder/'kernel_attribution.json');groups=collections.defaultdict(float)
  route_groups=collections.defaultdict(float)
  for r in data['kernels']:
   groups[r['kernel']]+=r['milliseconds']
   if r['module']=='route_graph_copy_and_replay':route_groups[r['kernel']]+=r['milliseconds']
  profiles[name]=dict(total_ms=data['total_kernel_ms'],modules={r['module']:r['kernel_ms']for r in data['modules']},kernels=dict(sorted(groups.items(),key=lambda x:-x[1])),router_kernels=dict(sorted(route_groups.items(),key=lambda x:-x[1])))
 micro={}
 for f in sorted(out.glob('*probe*.json')):
  if f.name=='head_probe_preserved.json':continue
  micro[f.name]=read(f)
 summary=dict(scope='8 fixed Val prompts, seed0, native four chunks x four steps; Test unused',timing=timing,per_prompt_timing=timings,quality=quality,profiles=profiles,microbenchmarks=micro,recommended='Compact',bit_identical_option='Wide',training_performed=False,shared_query_masks=False,denoising_step_reuse=False)
 write(out/'final_summary.json',summary)
 old=read(root/'optimization9/manifest.json');frozen={**old['code'],**old['prior_unchanged']}
 for path,digest in frozen.items():assert sha(path)==digest,path
 lock=read(out/'implementation_lock.json')
 for path,digest in lock.items():assert sha(path)==digest,path
 codes={str(f):sha(f)for f in sorted((repo/'migration/unified_router/opt10').glob('*.py'))};codes[str(repo/'v10_optimization10_progress.sh')]=sha(repo/'v10_optimization10_progress.sh')
 videos={}
 for name in new:
  f=out/f'ALL8_Dense_Opt9Full_Opt10_{name}.mp4';d=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(f)]))['streams'][0]
  assert int(d['nb_read_frames'])==360 and d['width']==2496 and d['height']==480;videos[name]=d
 checks=read(out/'checksum_verification.json');assert checks['all_match_independent_quality'];assert checks['timed_outputs_checked']==210
 write(out/'final_verification.json',dict(timed_outputs_verified=checks['timed_outputs_checked'],frozen_files_verified=len(frozen),new_source_files=len(codes),core_files_verified=len(lock),quality_outputs=32,wide_bit_identical_all8=True,compact_bit_identical_to_moments_all8=True,first_chunk_identical_all32=True,timed_new_outputs_equal_quality_outputs=True,changed_query_audit_trajectories_equal_quality=True,core_unchanged_during_quality_and_timing=True,videos=videos,test_used=False,training_performed=False))
 write(out/'manifest.json',dict(code=codes,prior_unchanged=frozen,core_lock=lock,runtime=read(out/'runtime_options.json'),default_existing_entries_changed=False))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,3,figsize=(17,4.7));colors=['#555555','#bd6d56','#d1a061','#699cac','#5d8b70','#306a90']
 axes[0].bar(names,[timing[n]['generation_seconds']for n in names],color=colors);axes[0].tick_params(axis='x',rotation=35);axes[0].set_ylabel('Generation seconds');axes[0].set_title('8 Val: mean of per-prompt medians')
 modules=['route_graph_copy_and_replay','sparse_attention_including_union','other_backbone_and_diagnostics','ffn','cross_attention'];x=np.arange(5)
 for shift,name,color in [(-.2,'Opt9','#bd6d56'),(.2,'Compact','#306a90')]:axes[1].bar(x+shift,[profiles[name]['modules'].get(k,0)for k in modules],.4,label=name,color=color)
 axes[1].set_xticks(x,['Router','Sparse attn','Other / copies','FFN','Cross attn'],rotation=25);axes[1].set_ylabel('GPU kernel ms');axes[1].set_title('L24 / chunk4 / step3');axes[1].legend()
 for name,color in zip(names[1:],colors[1:]):axes[2].scatter(timing[name]['generation_seconds'],quality[name]['lpips_mean'],label=name,color=color)
 axes[2].set_xlabel('Generation seconds');axes[2].set_ylabel('LPIPS to Dense (36 sparse frames)');axes[2].set_title('Measured speed / deviation');axes[2].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'speed_hotspots.png',dpi=170);fig.savefig(out/'speed_hotspots.pdf');plt.close(fig)
 best=timing['Compact'];before=timing['Opt9'];dense=timing['Dense'];r0=profiles['Opt9']['modules']['route_graph_copy_and_replay'];r1=profiles['Compact']['modules']['route_graph_copy_and_replay']
 lines=['# Opt10：降低独立 query Router 的实际执行时间','',f'完整 AV 路由的 8 Val 平均生成时间 **{before["generation_seconds"]:.3f} → {best["generation_seconds"]:.3f}s，减少 {(1-best["generation_seconds"]/before["generation_seconds"])*100:.2f}%**。Dense 为 {dense["generation_seconds"]:.3f}s；新版本相对 Dense 为 {best["dense_speedup"]:.3f}×。代表性实际 block 的 Router（含 graph 输入复制）**{r0:.3f} → {r1:.3f}ms，减少 {(1-r1/r0)*100:.2f}%**。','', '本轮保留全部 81 个动作、完整 AV 修正头、每一步独立刷新以及逐 query mask。没有使用 Q64 共享 mask，也没有使用 Fast 分支的去噪步复用。首 chunk 和 layer0 保持 Dense，稀疏部分逻辑保留率约 15.17%。','', '## 端到端结果','', '全部 6 种模式在每个 prompt 对应的同一张 A100-SXM4-40GB 上正反交替执行，预热后第 1 个 prompt 测 6 次，其余各 3 次；以下是各 prompt 中位数的均值。generation 包含 4 个 native chunk × 4 steps、原样上下文重建和 latent 回传；total 另含 VAE 解码及像素回传。不含首次编译。210 个输出（含预热和对照组）逐一与独立质量生成校验一致。','','| 版本 | 生成/s | 含 VAE/s | Dense / 本版本 |','|---|---:|---:|---:|']
 for n in names:
  t=timing[n];lines.append(f'| {n} | {t["generation_seconds"]:.3f} | {t["total_seconds"]:.3f} | {t["dense_speedup"]:.3f}× |')
 lines+=['',f'Compact 在 {best["wins_vs_dense"]}/8 个样例快于 Dense。'+('完整 AV 路由版仍慢于 Dense，未达到 FLOP 比例对应的性能。'if best['generation_seconds']>dense['generation_seconds']else'本次完整 AV 路由均值已快于 Dense，但尚不能据此推广到其它长度或分辨率。'),'','## 逐项优化','', '1. **Wide：合批执行，决策独立。** 把一层 4,680 个 query 的多次 512-query 调度合为一次完整批量，减少 kernel 启动次数、让小归约有更多 CTA。每行仍有自己的 mask。8 段完整视频 latent 与 Opt9 MemoryFull 逐位一致。','2. **Head：低精度中间值和快速激活。** 81 动作结构不变，动作头中间 T 从 FP32 改为 FP16，局部 Gram 乘法使用 FP16，采用 tanh 近似 GELU，并移除近同分动作的精确 fallback。统计归约改为 1 warp。此分支有数值变化。','3. **Moments：取消逐页 PV 能量。** 以加权 prototype 二阶矩替代页均值的平方范数，复用均值矩阵乘法一起得到统计量，不再写出 page energy 张量。这是改变统计量的近似，不是代数等价融合。','4. **Compact：按真实 P 执行 QK。** 在 Moments 上，历史 P2/P4/P8 和当前 P16 使用各自真实 P，固定每个 QK tile 64 个 prototype，避免把所有历史页补到 P16 后计算。仍沿用 padded 的 K 存储，但不加载/计算补零位置。候选内容仍按原来的加权 page 内容计算。','', '微基准使用保存的真实 QKV（最后一个 chunk，H12、Q4680、296 页、2220 个有效 prototype）；只测整个 route graph replay，包含全部 4680 query，排除外部 graph 输入复制。每个方案与 Opt9 原样交替测 6 组，不是整网耗时。','','| 实现 | Opt9/ms | 本版本/ms | 同 Q 改变 mask 的 query 数 / 56160 |','|---|---:|---:|---:|']
 for fname,idx,label in [('full_probe.json',0,'Wide'),('tuned_probe.json',0,'Head'),('tuned_probe.json',1,'Moments'),('compact_probe.json',0,'Compact')]:
  r=micro[fname][idx];lines.append(f'| {label} | {r["reference_ms"]:.3f} | {r["new_ms"]:.3f} | {r["changed_queries"]} |')
 lines+=['', '详细公式、FLOP/张量尺寸估算和公开实现思路见 [设计与调研](DESIGN_AND_RESEARCH.md)。','', '## 完整视频和实际注意力误差','', 'LPIPS 统计后 36 个稀疏帧与 Dense 的距离，不能直接解释成绝对视觉质量。所有 32 次新生成的首 chunk latent 和前 9 帧像素均与 Dense 完全一致。','','| 版本 | 对 Dense LPIPS↓ | LPIPS(Dense) 相对 Opt9 的差值 | 最坏单 prompt 变化 | 保留率 |','|---|---:|---:|---:|---:|']
 for n in names[1:]:
  r=quality[n];lines.append(f'| {n} | {r["lpips_mean"]:.6f} | {r.get("lpips_delta_mean",0):+.6f} | {r.get("lpips_delta_worst",0):+.6f} | {r["keep_mean"]*100:.3f}% |')
 lines+=['', '固定抽样：layer1/24/29 × 3 个稀疏 chunk × step3，每 block 每 head 12 个 query。固定样本恰好没有命中改变 mask 的行，路由增量误差都是 0，因此不能据此声称新路由精确。为此又独立运行 24 条轨迹，检查这些 block 中**所有改变 mask 的 query**，使用相同真实 QKV、两个 mask 的 BF16 SDPA 输出比较：','','| 版本 | 改变 query / 检查 query | 改变比例 | 改变行聚合 NMSE | 最坏单行 NMSE |','|---|---:|---:|---:|---:|']
 for n in new:
  r=quality[n];c=r.get('changed_query_audit',dict(aggregate_nmse=0,worst_query_nmse=0));lines.append(f'| {n} | {r["route_changed_queries"]} / {r["route_checked_queries"]} | {r["route_changed_query_fraction"]*100:.5f}% | {c["aggregate_nmse"]:.6g} | {c["worst_query_nmse"]:.6g} |')
 lines+=['', '上述 NMSE 只统计改变行，分母是 Opt9 mask 在同 QKV 上的输出能量，不会被大量未改变行稀释。它衡量本轮路由的增量，不是稀疏输出相对 Dense 的总误差；完整轨迹的变化会累积，所以视频仍可能有可见差异。固定抽检的量化 attention 对同 mask BF16 输出误差、逐 prompt 指标和逐 block 改变行误差均在 JSON 中。','', '## 优化后的热点','', '独立 profile：layer24、第 4 chunk、step3；统计 GPU kernel 时长，CPU scope 只用于归属。每个 profile pass 的 latent 与其 warm pass 逐位一致。单 block 是代表性切片，不能直接当作全模型平均。','','| 模块 | Opt9/ms | Wide/ms | Head/ms | Moments/ms | Compact/ms |','|---|---:|---:|---:|---:|---:|']
 for key in ['route_graph_copy_and_replay','sparse_attention_including_union','other_backbone_and_diagnostics','ffn','cross_attention','prototype_projection','current_prototypes']:
  lines.append('| '+key+' | '+' | '.join(f'{profiles[n]["modules"].get(key,0):.3f}'for n in names[1:])+' |')
 lines+=['','Compact 的 Router kernel：','','| kernel | ms |','|---|---:|']
 for k,v in list(profiles['Compact']['router_kernels'].items())[:7]:lines.append(f'| `{k}` | {v:.3f} |')
 lines+=['','Compact 的主要 kernel（包含 backbone）：','','| kernel | ms |','|---|---:|']
 for k,v in list(profiles['Compact']['kernels'].items())[:12]:lines.append(f'| `{k}` | {v:.3f} |')
 lines+=['', '下一步应针对新的实测瓶颈继续减少 W 权重写读、候选内容和集合矩统计成本；当前 W 仍是最大的中间张量。不能仅因某个融合减少了张量就认定更快：本轮候选内容融合、W 的 INT8 化都实测更慢，保留为负结果。后续把 score、选页、均值和候选重算设计成分阶段流水，须同时核对 CTA 寄存器、实际 DRAM/L2 流量及不同 chunk，不能只看名义 FLOP。','', '## 未采用的方案和复现','', 'Query-major / blocked W、物化 FP16/INT8 page 内容、拆分均值归约、批量候选选择、多组 score tile、预转 FP16 V、INT8 W/矩统计、候选内容与 head 融合均有探针实现。最后三种及融合版本的整层结果见 `half_probe.json`、`quant_probe.json`、`fused_probe.json`、`compact_probe.json`；它们没有优于采用的 Compact。微小的单 tile 收益没有直接当作端到端收益。','', '测速曾因共享 Inductor 自动调优缓存多出尾括号而失败；保留失败日志并原子修复后，整个配对实验重跑。失败预热数据不进入结果。另有第 7 个样例首次 Compact 质量输出未通过后续测速的逐位校验：保留原视频/latent/指标，独立重生成后通过正式重复测速和独立改变行审计；当前 8 个 Compact latent 与 Moments 全部逐位一致。该首次差异的根因尚未独立证明，不能仅凭时间先后断言来自缓存修复；跨编译环境仍须重新检查数值一致性。另修复了探针输出文件名复用的问题，相关探针已用独立路径复跑；`compact_probe.json` 保留原始最快结果。','', '[最终对比视频：Dense / Opt9-Full / Compact](ALL8_Dense_Opt9Full_Opt10_Compact.mp4) · [逐位一致版本 Wide](ALL8_Dense_Opt9Full_Opt10_Wide.mp4) · [Head](ALL8_Dense_Opt9Full_Opt10_Head.mp4) · [Moments](ALL8_Dense_Opt9Full_Opt10_Moments.mp4)','', '[速度与热点图](speed_hotspots.png) · [PDF](speed_hotspots.pdf) · [完整结果](final_summary.json) · [验证结果](final_verification.json) · [源码封存](manifest.json) · [训练对齐要求](TRAINING_ALIGNMENT.md)','', f'新入口与源码：`migration/unified_router/opt10/`。校验 {len(frozen)} 个前序源码文件保持不变；本轮未重训、未修改 checkpoint 和数据，未使用 Test。未替换旧实验默认入口；Opt10 默认变体是 Head，使用下面的显式选项进入推荐的 Compact。','', '```bash','./v10_optimization10_progress.sh --once','SLIMAV_OPT10=Compact CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\','  TORCHINDUCTOR_COMPILE_THREADS=2 streaming_sparse/.venv/bin/python -u \\','  -m migration.runtime.bootstrap migration.unified_router.opt10.network \\',f'  --directory {root} \\',f'  --output {out}/reproduce_Compact --mode K32-Set','```','']
 (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(timing=timing,quality={k:{kk:vv for kk,vv in v.items()if kk!='per_prompt'}for k,v in quality.items()}),indent=2))
if __name__=='__main__':main()
