"""Produce reviewable same-video tier, implementation, and policy tables."""
import json,statistics
from pathlib import Path
from .weights import ROOT

def main():
 root=ROOT/'artifacts/video_runtime_v1/tiers_m16_v1';load=lambda p:json.loads(p.read_text())
 names=['fast_q64','full_q64','fast_m16','full_m16','last_m16','full_g2_m16']
 results={n:load(root/n/'result.json') for n in names};decode={n:load(root/n/'decode.json') for n in names}
 quality={n:load(root/(n+'_quality.json')) for n in ['fast_m16','full_m16','last_m16','full_g2_m16']}
 route=load(ROOT/'artifacts/video_runtime_v1/route_m16_v1/result.json')
 rep=load(ROOT/'artifacts/video_runtime_v1/vector_g2_v1/repeat/stability.json')
 summary=dict(performance={n:dict(generation_s=x['median_seconds'],seconds=x['seconds'],vae_s=decode[n]['median_seconds'],total_s=decode[n]['generation_plus_decode_seconds'],stages=x['stages'],counters=x['runtime_stats']['counters']) for n,x in results.items()},quality=quality,route=route,vector_repeat=rep,
 full_increment_before_s=results['full_q64']['median_seconds']-results['fast_q64']['median_seconds'],full_increment_after_s=results['full_m16']['median_seconds']-results['fast_m16']['median_seconds'])
 (root/'summary.json').write_text(json.dumps(summary,indent=2))
 lines=['# 两档独立路径、M16入口、G=2原型及末次修正预算','','本轮结束：部署M16入口，保留基础档与集合修正档。G=2有稳定但很小的收益，保留实验开关、默认关闭，结束原型，不扩参数搜索。末次step修正作为独立策略开关，默认仍为全部step。没有启用整块Graph或状态矩阵化。','','## 1. 配置和计时口径','','固定city_walk、seed701，9chunks/105帧/16FPS/768×512，history360（5个历史chunk）、tile72；Page64/Q64/R4、完整V均值+rank8、M16/B16、10%精算、16×32候选+no-op。全覆盖C_s17/Full_Pset_s17和既有配套权重不变。每次修正仍执行一次512动作评分；预算只决定哪些noisy step执行。','','GPU6同卡顺序测量，完整9chunk预热一次，3次生成取中位数；排除模型加载、编译、latent IO、编码。VAE单独预热测量。各模块另跑一次CUDA事件插桩，不能与无插桩总时间逐项严格相加。所有方案保存source快照、权重哈希、展开deployment。未重跑84质量清单，未跑15s/30s。','','## 2. 两档同口径速度及增量成本','','|实现/档位|生成/s|相对旧同档加速|VAE/s|生成+VAE/s|','|---|---:|---:|---:|---:|']
 labels={'fast_q64':'旧入口·基础档','full_q64':'旧入口·完整修正','fast_m16':'M16·基础档（部署）','full_m16':'M16·完整修正（部署）','last_m16':'M16·仅末次修正（策略实验）','full_g2_m16':'M16+G2·完整修正（原型）'}
 for n in names:
  x=results[n];base='fast_q64' if n.startswith('fast') else 'full_q64';speed='—' if n.startswith('last') else f"{results[base]['median_seconds']/x['median_seconds']:.4f}×"
  lines.append(f"|{labels[n]}|{x['median_seconds']:.6f}|{speed}|{decode[n]['median_seconds']:.6f}|{decode[n]['generation_plus_decode_seconds']:.6f}|")
 lines+=['',f"旧入口的Full_Pset增量为 **{summary['full_increment_before_s']:.6f}s**；M16入口增量为 **{summary['full_increment_after_s']:.6f}s**。基础档不是把完整档速度套到关开关版本：权重加载、缓存分配、在线Reader工作均有独立路径。",
 '', '|模块，独立插桩累计/ms|基础档M16|完整修正M16|增量/ms|仅末次修正M16|','|---|---:|---:|---:|---:|']
 for key in ['summary_and_encoding','route_and_correction','attention','clean_commit']:
  values=[results[n]['stages'][key]['milliseconds'] for n in ['fast_m16','full_m16','last_m16']]
  lines.append(f'|{key}|{values[0]:.3f}|{values[1]:.3f}|{values[1]-values[0]:.3f}|{values[2]:.3f}|')
 lines+=['','两档差额主要在route，其次为页面编码及clean commit。共同主干未在这四项完整覆盖，包含QKV/O投影、FFN、归一化等，不能把剩余时间解释为Python开销。',
 '', '## 3. 独立配置和fast-only路径', '', '`full_pset`决定是否需要集合修正，`mode`决定实现路径；`pset_steps=all/last`决定在线执行次数。`--no-full-pset`不加载Reader/Phi/B，不分配页面编码和投影基，不保留完整修正图；基础档评分和编码计数均为0。仅持久页面编码就省112,150,656字节（112.15MB），其余权重及图池另计，未把显存变化当作测速收益。',
 '', '常驻worker按不可变manifest判断是否包含c_pset；纯Dense+C+Risk清单不加载修正器，混合清单保留两档权重/图。每个job的identity写实际full_pset，队列级identity写进程能力。batch透传pset_steps；manifest可指定`--methods dense c_risk`。',
 '', '## 4. M16入口：收益、理论工作量和数值边界', '', '|档位，P432/tile72|Q64入口含准备复制/ms|M16入口含准备复制/ms|加速比|','|---|---:|---:|---:|']
 for full,v in route['tiers'].items():
  a=v['rows']['False']['with_input']['gpu_ms_median'];b=v['rows']['True']['with_input']['gpu_ms_median'];lines.append(f"|{'完整修正' if full=='True' else '基础档'}|{a:.6f}|{b:.6f}|{a/b:.4f}×|")
 lines+=['','同一捕获P432输入，CUDA Graph 50次中位数，测量包括按生产布局选择/转换Query和复制到route输入。前次隔离记录基础档0.9001→0.8499ms、完整档4.3581→4.2445ms，最新复测见上表；基础档绝对时间有波动，最终收益以同卡短视频为准。',
 '', '容器28.312→7.078MB，减少21.234MB，复制读写至少减少42.467MB；按1266.396GB/s参考约0.0335ms，另省整Q64转换。选择probe也有成本。编译器因输入形状改变会调整融合和归约，完整route收益不能全归因于copy。Attention仍持有原Q64，probe数、offsets和顺序均不变。',
 '', f"基础档隔离Risk最大绝对差0.125，相对最大值{route['tiers']['False']['risk_relative_max']:.3g}、NMSE {route['tiers']['False']['risk_nmse']:.3g}；LUT、mask、logits、统计一致。完整档该捕获事件5个输出逐位一致。短视频全程接受交换计数696785→696781，不能声称所有动作逐位一致；两档最终latent/RGB均逐位一致、LPIPS0、RGB差0，基线也与既有84清单对应视频逐位一致。这是输入布局影响编译后的FP32舍入路径，没有改变数学公式或新增低精度近似。没有对未捕获中间值做全面逐操作归因。", '', '## 5. vector-only G=2：停止于一个原型', '', '保留原scalar、完整283.116MB WORK、S128和FP32页面归约树。每块处理同head两个相邻probe，逐段读取同一响应系数/mean/detail给两行使用；不生成[G,S,16]的全套中间数组，也不物化全部响应。现有warps/向量宽度不变。',
 '', '|项目|G1|G2|结论|','|---|---:|---:|---|']
 g1=statistics.median(v['1']['route']['gpu_ms_median'] for v in rep);g2=statistics.median(v['2']['route']['gpu_ms_median'] for v in rep)
 rr1=statistics.median(v['1']['reader']['gpu_ms_median'] for v in rep);rr2=statistics.median(v['2']['reader']['gpu_ms_median'] for v in rep)
 lines += [f'|Reader，3轮交替复测中位数/ms|{rr1:.6f}|{rr2:.6f}|{rr1/rr2:.4f}×|',f'|完整route，同上/ms|{g1:.6f}|{g2:.6f}|{g1/g2:.4f}×|','|首轮vector profile/ms|1.370388|1.331122|约1.0295×|','|寄存器/线程|255|255|达到高寄存器压力|','|Triton n_spills|0|20|G2新增spill|',f"|同卡视频生成/s|{results['full_m16']['median_seconds']:.6f}|{results['full_g2_m16']['median_seconds']:.6f}|{results['full_m16']['median_seconds']/results['full_g2_m16']['median_seconds']:.4f}×|"]
 lines+=['','G2模块隔离沿用捕获的Q64入口（仅比较向量kernel，未混入M16入口变化）。直接Reader三个输出、路由五个输出逐位一致；组合M16短视频相对M16/G1验收见full_g2_m16_quality.json。页面静态向量整层3.484MB，每行独立读取的逻辑量4.013GB；G2源码共享可将这部分理想逻辑读取减半，但不能当作HBM流量减半。WORK、输出流量、Query相关运算不变；spill和寄存器压力抵消了大部分潜在收益。',
 '', '结果是Reader约1.6%、route约2%、生成约0.10%的收益，远非2×。保留`--reader-vector-g2`用于复现，默认关闭。本轮到此收敛，不扩G/warps/寄存器搜索，也不启动直接状态矩阵化。',
 '', '## 6. 独立推理预算实验：仅最后一次noisy step', '', '四个noisy step索引0–3，last只在step3修正；step4始终为clean commit。配置需要Reader状态与当前step执行修正分开：所有clean commit仍编码历史，当前noisy页仅在即将执行修正时编码。step0–2走基础图，step3走完整图，graph key已有full_pset分支隔离。',
 '', '|预算|评分事件|编码head-pages|生成/s|相对完整档速度|','|---|---:|---:|---:|---:|']
 for n in ['fast_m16','last_m16','full_m16']:
  x=results[n];c=x['runtime_stats']['counters'];lines.append(f"|{n}|{c[1]}|{c[3]}|{x['median_seconds']:.6f}|{results['full_m16']['median_seconds']/x['median_seconds']:.4f}×|")
 lines+=['','在线评分701568→175392，恰好减少75%；clean commit仍261次。生成不是减少75%，仍有基础路径、编码维护和共享主干。本策略不调整候选数/阈值/权重，没有训练门控，不能列为同算法实现提速。',
 '', '|同一短视频相对Dense，[21,101)帧|LPIPS↓|SSIM↑|PSNR/dB↑|','|---|---:|---:|---:|']
 for n in ['fast_m16','last_m16','full_m16']:
  q=quality[n]['vs_dense'];lines.append(f"|{labels[n]}|{q['lpips']:.6f}|{q['ssim']:.6f}|{q['psnr']:.3f}|")
 lines+=['','Dense像素复用既有84清单中city_walk_s701参考，不重新生成/解码。LPIPS(AlexNet)/SSIM在256×384，PSNR原分辨率。last的LPIPS略优于完整档，但SSIM/PSNR更低；这只是一个固定短视频，不能证明末次step最有效或保留普遍质量收益。last对完整档latent NMSE约0.3583、RGB MAE约15.35/255，是推理策略变化，不属于优化舍入误差。保留all为默认。四列视频为Dense、基础、末次修正、完整修正：[预算并排视频](budget_comparison.mp4)。',
 '', '## 7. Timeline与下一步边界', '', 'GPU2/3分别捕获已预热chunk8，不开block_graph。基础档GPU区间973.37ms、union busy945.59ms、gap27.78ms（2.85%）；完整档1407.62/1370.91/36.71ms（2.61%）。最大gap约0.29ms。profile本身改变主机调度，gap不是可承诺回收的Python时间，但没有大块空隙支持本轮投入整块Graph。完整trace和聚合保存在timeline目录。不同GPU的这两份trace不用于正式两档加速比。',
 '', 'P2已经结束；直接状态矩阵化可能改变归约顺序并引入加权特征开销，作为后续工程保留，不在本轮启动。',
 '', '## 8. 实现与复现位置', '', '- [deployment.py](../../../local_experiments/video_runtime_v1/deployment.py)：独立full_pset、pset_steps、route_m16及G2配置；当前M16开、G2关、all默认。','- [weights.py](../../../local_experiments/video_runtime_v1/weights.py)：include_reader=False绕过Reader/Phi/B加载。','- [runtime.py](../../../local_experiments/video_runtime_v1/runtime.py)：可选持久编码、correct step分支、M16选取、clean commit维护。','- [pipeline_graph.py](../../../local_experiments/video_runtime_v1/pipeline_graph.py)：input_m16入口、保留原Q64对照；key包含分支与Query行数。','- [reader_vector_g2.py](../../../local_experiments/video_runtime_v1/reader_vector_g2.py)、[reader_reduce.py](../../../local_experiments/video_runtime_v1/reader_reduce.py)：唯一G2向量原型，custom-op显式vector_group及对应fake签名。','- [worker.py](../../../local_experiments/video_runtime_v1/worker.py)、[queue.py](../../../local_experiments/video_runtime_v1/queue.py)、[manifest.py](../../../local_experiments/video_runtime_v1/manifest.py)、[batch.py](../../../local_experiments/video_runtime_v1/batch.py)：manifest能力决定权重加载，逐任务实际档位identity、策略透传。','- `check_route_m16.py`、`check_vector_g2.py`：相同捕获事件隔离测试；后者只G1/G2及交替复测。','- `run_tier_video.py`、`benchmark_hotspots.py`：同卡顺序短视频计时；`quality_tiers.py`：优化误差与策略相对Dense指标；`profile_tier_timeline.py`：两档timeline；`report_tiers.py`：本报告生成。','', '统一使用 `/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.video_runtime_v1.<模块>`。视频复现例如`CUDA_VISIBLE_DEVICES=6 ...run_tier_video --names full_q64 fast_q64 full_m16 fast_m16 last_m16 full_g2_m16`；输出目录固定，重跑先另存既有证据，不覆盖本次报告。基础部署命令见`local_setup/CURRENT_VIDEO_BASELINE.txt`。', '', '原始证据：[summary.json](summary.json)、每方案result/decode/source、*_quality.json；[M16隔离](../route_m16_v1/result.json)、[G2首轮](../vector_g2_v1/result.json)、[G2交替复测](../vector_g2_v1/repeat/stability.json)、[寄存器与spill](../vector_g2_v1/kernel_resources.json)。']
 (root/'MODULE_DISTANCE.zh.md').write_text('\n'.join(lines)+'\n')
 print('wrote',root/'MODULE_DISTANCE.zh.md')
if __name__=='__main__':main()
