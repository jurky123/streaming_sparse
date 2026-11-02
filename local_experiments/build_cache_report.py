"""Generate the Markdown report from completed, audited research artifacts."""
import csv
import json
import math
from pathlib import Path
import statistics


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    root=Path('artifacts/cache_research_20261029').resolve()
    audit=json.loads((root/'audit.json').read_text());assert audit['passed']
    screen=json.loads((root/'screen/summary.json').read_text())
    lookup={(r['group'],r['mode']):r for r in screen}
    video=json.loads((root/'analysis/video_summary.json').read_text())['rows']
    extra=json.loads((root/'analysis/followup_summary.json').read_text())['rows']
    allrows=video+extra
    labels={'dense':'Dense','spatial':'空间分块','k':'K-only','kv1':'Joint-KV','k_tail':'K + 全层补偿','kv1_tail':'KV + 全层补偿','k_guard':'K + 第 8 层补偿'}
    methods=list(labels)
    means={m:{k:statistics.mean(r[k] for r in allrows if r['method']==m) for k in ['brightness_step_mean_pp','dark_dip_max_pp','chroma_drift','flow_mean','dino_adjacent','latent_nmse','generation_seconds','preprocess_gpu_seconds']} for m in methods}
    bench=json.loads((root/'microbenchmark_interleaved.json').read_text())['summary']
    diag=json.loads((root/'diagnostic/layer_selection.json').read_text())
    rope=json.loads((root/'rope_statistics_probe.json').read_text())
    reductions={g:100*(1-lookup[g,'var_tail']['nmse']/lookup[g,'mass']['nmse']) for g in ['k','kv1']}
    sections=[]
    sections.append(f'''# 自回归视频稀疏注意力：调研与验证结果

2026-10-29。**已完成 18 条新的 30 秒视频、24 组真实 QKV 消融，以及全层失效诊断。**另复用 3 条同输入 dense 视频作为参考。报告为 Markdown，图为 PNG/PDF，视频为 MP4。

这轮最有价值的结论是：**重预处理值得做，但 raw Joint-KV 聚类不是稳定改进；应围绕输出重构、分布校准和缓存生命周期设计。**全层 tail 补偿在初始抽样评估中降低误差，却让完整视频明显变灰、变糊，当前版本应淘汰。选择性补偿只是小范围后续验证，也没有证明全面优于基线。

- 你的第一个想法成立于 **clean 历史 KV**：108 次逐元素检查均未变化，采集后的 latent 与原 dense 完全一致。当前 noisy KV 每步会变；FIFO 历史最多复用 20 次，不能按无限寿命摊销。
- 你的第二个想法需要加上目标约束：相同预算下，K-only / Joint-KV 的抽样输出 NMSE 为 {lookup['k','mass']['nmse']:.5f} / {lookup['kv1','mass']['nmse']:.5f}。单纯加入 V 没有获益。
- tail/方差补偿将相应抽样 NMSE 降低 {reductions['k']:.1f}% / {reductions['kv1']:.1f}%，**但这是抽样层的局部结果，不是视频提升**。全层诊断与闭环生成推翻了直接推广的判断。
- 下一轮优先考虑：**按层/头校准的输出重构目标、低秩 K-V 交叉矩、写入时重排和按驻留寿命分配预处理成本**。后两类机制推导和已有工作边界见 [详细调研](research_notes.zh.md)。

## 实验约束

同一 Self-Forcing EMA checkpoint、同一保存的初始/过渡噪声与文本 embedding，512×768、16 fps、41×3 latent frames，裁为 480 RGB 帧。三个 prompt：sailboat、skateboard、dog，各 1 个 seed。所有新视频保持原生 absolute RoPE 和 FIFO，没有新增 rebase 对比组。

持久 self-KV 统一为 **5,096,079,360 bytes = 4.74609375 GiB**，包括 5 个历史 chunk 和当前 chunk scratch。分组索引另占最多 30.586 MiB；没有额外持久 summary KV，也没有长期重排 KV 副本。算子临时工作区和峰值显存另计。这个实验研究少算 attention，没有把完整历史 KV 压缩到 20%。

历史计算上限为 20%：普通臂稳态精确计算 72×64=4608 个历史 token；补偿臂为 60×64=3840，加所有 360 页 summary/方差的保守 720 token 等价，共 4560≤4608。当前 4608 个 token 全算。因此“20% 历史”不是“20% 总 attention”。路由、重排、归约和 launch 的真实代价仍由计时报告，token 等价不是精确硬件 FLOPs。

第一个 chunk、layer 0、clean-context 写缓存 pass 全 dense，其他 29 层的四个 denoise step 使用各自策略。新原型调用 Light Forcing 的 sparse forward kernel，但分组、筛选和补偿由本地代码实现，**不是 SlimAV 或 Light Forcing 官方路由复现**。

[先验协议](protocol.zh.md) · [视频协议](video_protocol.zh.md) · [失败后诊断协议](diagnostic_protocol.zh.md) · [完整审计](audit.json)

## 真实 QKV 消融

抽样覆盖 3 场景×3 个 chunk×3 层×3 heads×4 去噪步×64 query，共 20,736 个 query 实例。24 个配置对同一批实例评估。query tile 为 64，mask 根据 tile 均值选择；不存在用未来 query 建簇。这里的多层/多头观察不是独立 prompt 样本。
''')
    rows=[]
    for g in ['spatial','k','kv025','kv1']:
        rows.append([g]+[f"{lookup[g,m]['nmse']:.5f}" for m in ['mass','value','tail','var_tail','hysteresis','oracle_mass']])
    sections.append(table(['分组','Mass','V 残差筛选','均值 tail','方差 + tail','Mask 惯性','精确 mass 诊断'],rows))
    sections.append('''
NMSE 为 `Σ||o_sparse-o_dense||² / Σ||o_dense||²`，先在各 head/query 集内计算，再平均。精确 mass 列读取 dense 权重，只作诊断，**不能作为可部署方法或全局最优子集 oracle**。KV025/KV1 是中心化、RMS 归一化后 V 的平方距离权重 0.25/1。

K-only 的 mask churn（1−Jaccard）由 0.1412 降为 0.0895；Joint-KV 由 0.1430 降为 0.0903，约减少 37%。这是相邻去噪步固定 query 位置的离线结果，尚未证明可以减少视频闪烁。V 残差筛选对内容分组有小幅局部收益，对空间分块反而稍差。

![逐场景展示 attention 输出 NMSE 和输出均值偏移；每条线是一个 prompt，不能据此推断视频质量。](analysis/mechanism.png)

[全部行 CSV](screen/metrics.csv) · [逐场景汇总](analysis/mechanism_by_case.csv) · [layer/step 分解](analysis/layer_step.png) · [数值与不变性核验](prototype_checks.json)

## 全层检查揭示的失败

帆船的全层补偿视频出现快速泛灰、模糊和运动衰减。补查全部 29 个稀疏层、12 heads、3 chunks、4 步后，Joint-KV 的第 3 层平均 NMSE 从约 0.018 增至 0.204，约 11 倍；第 28 层则从 0.00886 增至 0.228，约 26 倍。K-only 第 28 层也从 0.00780 增至 0.184。tail 概率误差的 Joint-KV 第 28 层 95 分位约 0.984。这些层不在最初的 4/14/24 层抽样里。

因此，本轮不能以抽样平均 NMSE 支持全层替换。Gaussian/对角方差可能在个别 head 失真；`mu_V` 还忽略了 query 对簇内 V 的条件选择；dense 轨迹校准又不能完全覆盖稀疏 rollout 后的分布。最终核查另修复了离线参考计算在极端分数下的下溢；旧记录保存在 `diagnostic_before_stability_fix/`，表和图全部使用重算后的有限值。视频 kernel 本来使用稳定的 logsumexp，视频无需重跑，层准入名单也未改变。

固定准入标准只让 K-only 的第 8 层通过，Joint-KV 无层通过。后续仅第 8 层补偿，其他层回到原 K-only；sailboat 是校准场景，skateboard/dog 未参与这次层选择。没有反复调阈值挑视频。

![全部层的输出误差与 tail 概率误差，显示初始抽样遗漏的异常层。](analysis/layer_diagnosis.png)

[全层原始数据](diagnostic/metrics.csv) · [准入规则和层名单](diagnostic/layer_selection.json)

## 闭环视频结果

下面是三个场景的均值。视觉判断基于 0/5/10/15/20/25/29.94 秒的固定采样图，亮度曲线来自全部 480 帧。亮度/色度/光流均是描述量；低光流可能意味着画面失去运动，低色漂可能只是输出变灰。DINO 是全图 cosine，不是完整 VBench；latent NMSE 也不是整体质量分数。**全层补偿臂虽然某些数值较低，采样画面明显失败。**
''')
    sections.append(table(['方法','亮度变化 pp','最大短暗降 pp','色度漂移','光流 px/s','DINO 邻秒','latent NMSE'],
        [[labels[m]]+[f"{means[m][k]:.3f}" for k in ['brightness_step_mean_pp','dark_dip_max_pp','chroma_drift','flow_mean','dino_adjacent','latent_nmse']] for m in methods]))
    sections.append('''
区间沿用前轮：RGB 帧 69–479。短暗降是低于中心 9 帧中值的亮度差；色度是全图平均 RGB 的归一化值相对开头漂移；光流在 192×128、4fps 上换算 px/s。内容、构图、光照与相机运动均会混淆这些指标。没有显著性检验或综合质量排名。

![六项描述指标按三个场景分别展示；不能将运动下降解释为更稳定的好视频。](analysis/video_metrics.png)

[逐场景主实验 CSV](analysis/video_metrics.csv) · [选择性补偿 CSV](analysis/followup_metrics.csv) · [原始逐帧与 DINO 数据](analysis/provenance.json)

选择性补偿避免了全层方案的灰雾式退化，但没有形成稳定优势：滑板最大短暗降从 K-only 的 2.025 pp 升到 3.746 pp；狗的 latent NMSE 从 0.793 升到 0.891。帆船的 latent NMSE 略降，采样图仍存在背景色块。三个场景中都不能仅凭某个改善指标决定采用该方案。

六宫格顺序：上排 Dense / 空间分块 / K-only，下排 Joint-KV / K 全层补偿 / KV 全层补偿。原视频均为 768×512；六宫格只缩小显示，无色彩增强。
''')
    for case in ['sailboat','skateboard','dog']:
        sections.append(f'''### {case}

[30 秒六宫格](video/{case}/comparison.mp4) · [K-only](video/{case}/k.mp4) · [Joint-KV](video/{case}/kv1.mp4) · [仅第 8 层补偿](followup/{case}/k_guard.mp4)

![{case} 六行时间采样图，所有方法使用相同输入、同一 KV 上限。](video/{case}/contact_sheet.png)

[Dense / K-only / 第 8 层补偿的时间采样图](followup/{case}/contact_sheet.png)
''')
    sections.append('''## 预处理能否回本

使用真实 12-head 层，Q=4608、历史 K=23040、D=128。24 轮交错执行顺序，记录同步 wall time 和 CUDA event；dense 使用与生成相同的四维融合 SDPA。下面是本地算子中位数，不是完整模型加速比。早期三头试验和顺序计时也保留，但不从它们外推完整层。
''')
    rows=[['Dense',f"{bench['dense']['wall_ms']:.3f}",'0',f"{bench['dense']['wall_ms']:.3f}"]]
    for name,pre in [('spatial',None),('k','k_preprocess'),('kv1','kv1_preprocess'),('k_tail','k_preprocess'),('kv1_tail','kv1_preprocess')]:
        p=0 if pre is None else bench[pre]['wall_ms'];s=bench[name]['wall_ms'];rows.append([labels[name],f'{s:.3f}',f'{p:.3f}',f'{s+p/4:.3f}'])
    sections.append(table(['方法','在线 ms','单 chunk 构建 ms','稳态在线 + 构建/4 ms'],rows))
    sections.append(f'''
这里每次 attention 同时使用 5 个 chunk，因此摊销为 `S + 5P/20`。本轮每个构建状态实际复用 4–20 次，平均 19 次。完整层上，普通 K/Joint-KV 构建有摊销空间；带补偿的在线成本已经偏高，不能宣称整体提速。

另做了“KV 已经按簇存放”的算子探针：K-only 在线由 {bench['k']['wall_ms']:.3f} ms 降到 {bench['k_preordered']['wall_ms']:.3f} ms。说明重复 gather 有明显代价。**视频原型仍缓存排列并逐调用 gather；预重排布局只完成了算子验证，未声称已集成进整段视频。**

[24 轮原始计时](microbenchmark_interleaved.json) · [早期完整层顺序计时](microbenchmark_fullhead.json) · [三头计时](microbenchmark.json)

生成端受共享 GPU 负载影响很大：选择性补偿与主实验使用不同物理 GPU/时段，dense 复用前轮记录。CSV 保留耗时和峰值显存，但这些端到端时间不能用于固定加速比。

## 一个额外可做点：统计摘要也要尊重位置几何

对真实 Q 和所有 K 施加相同的 temporal RoPE-pair 旋转，精确 attention 应保持不变。离线探针验证了这一点；对角 K 方差摘要却随坐标系改变。2×2 旋转对协方差可以保住这种不变性。
''')
    sections.append(table(['layer','精确输出旋转 NMSE','对角统计平均 log-mass 变化','2×2 统计平均变化'],
        [[r['layer'],f"{r['exact_attention_rotation_nmse']:.2e}",f"{r['diagonal_mean_log_mass_difference']:.4f}",f"{r['pair_blocks_mean_log_mass_difference']:.2e}"] for r in rope['records']]))
    sections.append('''
这是**共同坐标变换的数学探针，不是 anchor-only rebase 或新视频结果**。它说明缓存预处理不宜随意丢弃 RoPE 对内协方差；也不意味着 2×2 统计能解决 non-Gaussian tail 或缺失 K-V 交叉矩。[原始探针结果](rope_statistics_probe.json)

## 继续做什么，以及如何复现

建议先把目标从“更复杂的 KV 聚类”改成“**可缓存的、按层/头校准的 attention 输出近似**”，再选择其分组形式。优先级与具体可证伪判据、公式推导、现有方法的缺点和源码依据，见 [详细调研](research_notes.zh.md)。本轮验证足以排除直接全层启用当前 tail 补偿，但不足以宣称某种新方案优于 Light Forcing 等完整方法。

```bash
cd /data/chenjiayu/zhikun/streaming_sparse
CUDA_VISIBLE_DEVICES=7 bash local_setup/run_cache_research.sh
```

脚本复用上一轮保存的输入/权重，跳过已完成视频，依次执行捕获、消融、数值核验、视频、全层诊断、选择性 follow-up、完整层计时、评估和审计。结果集中在 `artifacts/cache_research_20261029/`。主要实现位于 `local_experiments/cache_grouping.py`、`cache_sparse_prototype.py`；数值、预算、首 chunk、固定噪声、41 chunks 与 480 帧均有运行时断言。

没有更改训练权重，也没有为结论挑掉失败视频。初始机制筛选属于探索；全层诊断是看到帆船失败后的 follow-up；其余两场景只对这次层选择留出，不是整项研究完全未观察过的数据。扩大 prompt/seed、用新场景确认、改善输出目标与 kernel 后，才适合讨论方法级质量或速度优势。
''')
    (root/'report.zh.md').write_text('\n\n'.join(sections)+'\n')
    print(root/'report.zh.md')


if __name__=='__main__':main()
