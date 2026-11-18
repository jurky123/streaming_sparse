"""Record module work, paired video timing and numerical acceptance."""
import json
from pathlib import Path
import torch
from .weights import ROOT

def main():
 root=ROOT/'artifacts/video_runtime_v1/grouping_dataflow_v1';video=root/'video';rows=[]
 for tier in ('base','intersection','step'):
  old=json.loads((video/(tier+'_old')/'result.json').read_text());ref=torch.load(video/(tier+'_old')/'latents.pt',weights_only=True).float()
  for mode in ('new','layout'):
   d=json.loads((video/(tier+'_'+mode)/'result.json').read_text());got=torch.load(video/(tier+'_'+mode)/'latents.pt',weights_only=True).float();delta=got-ref
   rows.append(dict(tier=tier,variant=mode,baseline_seconds=old['median_seconds'],seconds=d['median_seconds'],speedup=old['median_seconds']/d['median_seconds'],old_commit_ms=old['stages']['clean_commit']['milliseconds'],commit_ms=d['stages']['clean_commit']['milliseconds'],latent_equal=torch.equal(got,ref),latent_nmse=float(delta.square().sum()/ref.square().sum()),latent_max_abs=float(delta.abs().max()),counters=d['runtime_stats'].get('counters')))
 (root/'acceptance.json').write_text(json.dumps(rows,indent=2)+'\n')
 a=json.loads((root/'result.json').read_text());b=json.loads((root/'softmax/result.json').read_text());s=json.loads((root/'softmax_prototype.json').read_text())
 lines=['# 三档共有路径：分组数据流与softmax布局优化','','## 结论与验收','','固定city_walk_s701、9chunks/105帧、768×512、history360、tile72。base/intersection/step分别在GPU5/6/7同卡比较旧版、数据流版、连续softmax版；每次完整预热后重复5次，取中位数，单独插桩分解阶段。只报告生成，不含VAE/IO。','', '| 档位 | 改动 | 原生成/s | 新生成/s | 加速比 | latent逐位一致 | NMSE |','|---|---|---:|---:|---:|---|---:|']
 for r in rows:lines.append(f"| {r['tier']} | {r['variant']} | {r['baseline_seconds']:.5f} | {r['seconds']:.5f} | {r['speedup']:.4f}× | {r['latent_equal']} | {r['latent_nmse']:.6g} |")
 lines+=['','## 热点依据','','三档满窗口chunk8 profile：每chunk精算116次累计约88ms、补偿约51ms；clean分组29次累计约79.5ms，单次约2.74ms。分组内部的非连续维softmax约0.50ms，是其最大单kernel。CPU/CUDA的同名记录不可重复相加，原始trace按kernel事件归因。','', '## 改动一：删掉无用搬运','','硬分组外层5级、内层2级balanced排序，在最后一级之后不再使用特征x，只返回ids；删除各自最后一次x.gather。CUDA Graph的输出在当前forward内由summary/commit消费完成，下次group调用才覆写，因此直接返回图输出，无需复制kg/vg/order。参数只更新实际读取的bank/group_v/group_gate。graph必须持有page_order，避免外部索引缓冲被回收。初版漏持有引发device assert，已修复；最终四组回放及三档视频验证通过。','', f"删除两次特征gather逻辑读写28.312MB、输出clone读写57.508MB，总85.819MB/commit。按既有实测复制带宽1266.396GB/s折算约0.0678ms；这只是搬运参考。完整分组{a['medians']['old']['gpu_ms']:.4f}→{a['medians']['new']['gpu_ms']:.4f}ms，{a['speedup']:.4f}×，含输入复制、模型更新及输出处理。",'', '## 改动二：softmax沿连续维计算','','原score为[12,1,4608,32]，沿4608维softmax。改为transpose(-2,-1).contiguous()，沿最后一维softmax，再transpose返回。没有降精度、没有改温度/归一化公式、没有减少分组迭代；改变了浮点归约顺序。','',f"独立真实输入（layer15，chunk8 clean）：包含转置搬运{ s['old_ms']:.4f}→{s['new_ms']:.4f}ms，{s['speedup']:.2f}×。softmax最大绝对差{s['softmax_max_abs']:.3g}，特征最大差{s['feature_max_abs']:.3g}；该输入最终排序一致。完整分组{b['medians']['old']['gpu_ms']:.4f}→{b['medians']['new']['gpu_ms']:.4f}ms，{b['speedup']:.4f}×；四组原输入/移位输入/修改group_gate均输出一致。",'', '## 理论工作量与剩余空间','','score共有1,769,472个FP32元素。旧softmax只按输入读/输出写计14.156MB；新路径还显式转置读写14.156MB，共28.312MB。以1266.396GB/s折算分别0.0112/0.0224ms，均未计指数、归约通信和缓存行为，不能当可达到下限。新路径虽然增加搬运，却大幅减少低效的跨行归约成本，这才是收益来源。','', '分组两次FP32特征投影约0.906GFLOPs，七级balanced的4轮双matvec加最终投影约0.223GFLOPs，总约1.13GFLOPs，另有softmax、norm、稳定排序和gather。不得用大GEMM吞吐直接推导这类混合流程的可兑现加速。完整分组收益乘261次commit，才是短视频约可节省的量级；整个主干、Attention和路由都不因该改动消失。','', '## 代码与重现','','- [数据流及布局原型](../../../local_experiments/video_runtime_v1/grouping_dataflow.py)：balanced、group_features、GraphGrouping。','- [隔离测试](../../../local_experiments/video_runtime_v1/check_grouping_dataflow.py)：真实输入、输入及权重刷新、交错计时；`--softmax`测试布局版。','- [softmax单模块](../../../local_experiments/video_runtime_v1/check_grouping_softmax.py)：数值差与最终排序。','- [细分profile](../../../local_experiments/video_runtime_v1/profile_retained.py)、[输入捕获](../../../local_experiments/video_runtime_v1/capture_grouping.py)。','- [部署](../../../local_experiments/video_runtime_v1/deployment.py)、[三档入口](../../../local_experiments/video_runtime_v1/accepted_tiers.py)。','- [模块原始结果](result.json)、[布局模块结果](softmax/result.json)、[视频结果](acceptance.json)。','', '入口实验开关：`--grouping-dataflow`、`--grouping-softmax`（布局版包含数据流优化）。本轮不改变三档的Full_Pset预算或权重。完整视频数值不一致时保留开关并记录误差，不把局部排序一致推广到整个模型。']
 lines+=['', '## 最终部署决定与误差定位', '', '默认启用纯数据流优化；连续softmax保持默认关闭。数据流版三档latent逐位一致，布局版的基础/step档发生轨迹差异。以下误差相对未优化的同一档位，不是相对Dense，也不据此推断算法质量变差。', '', '| 档位 | PSNR/dB | SSIM | LPIPS | RGB MAE/255 | RGB逐位一致 |', '|---|---:|---:|---:|---:|---|']
 for tier in ('base','intersection','step'):
  q=json.loads((video/(tier+'_layout_quality.json')).read_text());v=q['quality'];lines.append(f"| {tier} | {v['psnr']:.4f} | {v['ssim']:.6f} | {v['lpips']:.6f} | {q['rgb_mae_255']:.6f} | {q['rgb_equal']} |")
 lines+=['', 'PSNR/SSIM/LPIPS按帧[21,101)；RGB差按全部105帧。完全相同视频的SSIM实现有约6e-7数值偏差、PSNR封顶100dB。', '']
 for tier in ('base','step'):
  audit=json.loads((root/(tier+'_order_audit.json')).read_text());changed=[r for r in audit['rows'] if r['different_token_positions']]
  lines.append(f"- {tier}：沿原版clean轨迹对261个相同输入逐个比对，{len(changed)}个事件排序不同；首个事件{changed[0] if changed else None}。")
 lines+=['', '误差类型：FP32归约顺序/内存布局引起的舍入差，经过硬排序和后续生成放大；没有引入低精度或近似softmax。单个layer15输入排序一致不足以覆盖全部clean事件，完整视频与261事件审计揭示了这一点。', '', '进一步kernel证据：原softmax仅12个CTA，grid=[12,1,1]、block=[32,32,1]，26寄存器/thread，4KB共享内存，profile估计occupancy约6%。下一项有依据的原型是保持每列32线程的累加及归约树，将32个特征列分给更多CTA；先确认与原算术一致，再测性能。它与本次改变归约树的连续布局方案不同。']
 (root/'MODULE_DISTANCE.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
