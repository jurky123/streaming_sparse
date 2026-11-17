# streaming_sparse 当前状况交接

**9d4d33d审查收尾已完成**：batch继承tile72/history360，透传Reader权重和local-assets；实际84个短视频任务及后处理全部完成，28组三列视频已保存。[质量报告](../../artifacts/video_runtime_v1/quality_batch_9d4d33d_v1/REPORT.zh.md)：Full_Pset在19/28组降低LPIPS，均值0.3181→0.3173，前段改善、后段略退化，seed差异明显。该清单冻结9d4d33d，仅叠加入口修正。P2仅试512/1024行，均逐位一致但无净收益，[结束不部署](../../artifacts/video_runtime_v1/p2_bounded_v1/REPORT.zh.md)。扫描直接消费FP32候选、内部FP64，消去ca转换；路由4.314→4.247ms（1.016×），生成10.551→10.493s，latent/RGB逐位一致，[验收并默认部署](../../artifacts/video_runtime_v1/scan_fp32_video_v1/MODULE_DISTANCE.zh.md)。以下为此前各轮历史记录；没有启动P3或15s/30s。

**本轮按ccb4621审查意见完成P0/P1**：[实现位置、模块距离与验收](../../artifacts/video_runtime_v1/storage_cached_video_v1/MODULE_DISTANCE.zh.md)。新增共享deployment.py，worker默认全覆盖C_s17/Full_Pset_s17、history360、tile72、全部已验收优化；实际常驻进程一次加载完成Dense/c_pset短视频，c_pset与旧验收benchmark逐位一致，独立后处理和Dense复用通过。三项P1：Reader FP32输出容器、CO/A保留既有BF16值、分母端点预组合。首次纯torch公式仍把V_a内联到512动作，补custom-op边界后兑现1552→576次logaddexp/probe。最终满窗口4.620→4.304ms（1.074×），生成10.772→10.522s、含VAE15.071→14.839s，latent/RGB逐位一致、LPIPS0。当前命令新增--reader-output32 --coeff-bf16 --endpoint-den --cached-endpoints，worker默认与最终验收配置一致。Reader输出203.489→101.745MB，但Reader只省约0.058ms；分母0.331→0.117ms。Reader参考已计输出写，0.447ms原口径修正为旧0.608/新0.527ms；不是可承诺收益。完整路由Risk数组部分编译配置有微差，LUT/动作/统计一致；直接Reader契约逐位，评分最大2.78e-17，14份E49、主导页和fallback通过。P2小WORK和P3向量跨Query未启动；没有跑15s/30s或新一批大规模质量评估。

**本轮已完成，按用户要求暂停优化目标**：[详细模块距离与验收报告](../../artifacts/video_runtime_v1/logmass_video_v1/MODULE_DISTANCE.zh.md)。本轮增加 --cached-logmass：FP64 R4局部log质量只计算一次，再作页面归一化，旧快精算/补偿不变。8组布局筛查＋216/288/360页补测；完整路由4.915→4.580ms（1.073×），捕获事件路由输出逐位一致。GPU3同卡9chunks短视频生成10.937→10.735秒（1.019×），含VAE15.245→15.043秒（1.013×）；latent/RGB逐位一致、LPIPS0、RGB差0。局部log质量最大2.27e-13为FP64计算次序差，无精度降低。部署log质量约0.560→0.223ms，实际两阶段payload参考0.189ms；Reader仍约1.938ms是后续热点。本轮结束后不再启动下一轮；恢复须用户指示。当前命令已加 --cached-logmass，通用开关默认关闭。

**当前基线新增 --projected-risk**：[Risk验收](../../artifacts/video_runtime_v1/risk_projected_video_v1/REPORT.zh.md)、[更新模块距离表](../../artifacts/video_runtime_v1/risk_projected_video_v1/MODULE_DISTANCE.zh.md)。继续使用用户指定旧快精算/补偿，保留原FP32 SGEMM投影，融合Risk特征和M16归约。216/288/360/432页隔离2.113–2.573×，候选池一致；完整路由4.996→4.911ms（1.017×）。GPU3同卡短视频生成11.027→10.948秒，含VAE15.336→15.261秒；latent/RGB逐位一致、LPIPS0、RGB差0。局部Risk有约3–5e-15 NMSE的舍入差。上一表Risk约0.875ms包含Reader共享log质量工作，本轮修正归因：Risk主要路径0.469ms、Reader log质量0.560ms。后者三遍扫描重复局部R4质量，是下一候选热点。24组筛查＋4窗口复验已完成。当前命令见local_setup/CURRENT_VIDEO_BASELINE.txt；通用实验开关仍默认关闭。

**当前优化基线新增--scan-exclude**：[正质量前后缀扫描验收](../../artifacts/video_runtime_v1/exclude_scan_video_v1/REPORT.zh.md)。保留用户选定旧快精算/补偿和A-only/split Reader；PAGE32/lazy关闭。16/32候选组不再物化K×K权重矩阵，使用FP64正质量前后缀汇总；移出最大质量页另以剩余最大值缩放，不作主导质量相减。四窗口评分1.124–1.166×、动作/no-op一致；完整路由5.305→4.997ms。GPU3单短视频生成11.310→11.033秒（1.025×），含VAE15.627→15.350秒（1.018×）；latent/RGB逐位一致、LPIPS0、RGB差0。14份E49及主导页测试通过，直接聚合覆盖K16/32和最大页头/中/尾1000-log差距。源码通用参考开关默认关闭，当前命令已显式加入--scan-exclude；未来优化应以此为对照。直接替换BMM或直接softmax/value融合均变慢，未采用。理论距离已细化：原BMM0.566GFLOPs的算术下限0.041ms过于乐观，311MB逻辑payload对应约0.246ms访存参考，实测0.372ms仅约1.51倍；新扫描两核约0.293ms。GPU扫描存在logK步，不能把概念线性扫描误作严格线性发射FLOPs。

最新分母/动作评分筛查完成：[20组隔离与编译证据](../../artifacts/video_runtime_v1/den_score_v1/REPORT.zh.md)。使用旧快Attention新基线实际捕获输入，融合最多约1.010×，单独预计算约1.002×，不推广、未运行本轮视频。14份E49和主导页稳定性案例通过；所有捕获样本动作/no-op一致但评分有微小舍入差。关键修正：当前Inductor已经将第一步logaddexp共享到221184行，后续7077888项只剩3次logaddexp；原先“可省686万次”的估计不适用于当前编译代码。源码证据已存compiler_evidence。FUSED_DEN/PRECOMPUTE_DEN默认false；当前基线仍旧快速精算/补偿＋A-only/split Reader，PAGE32/lazy关闭。

**当前部署基线已按用户要求切换精算/补偿为旧快实现**：[验收与配置](../../artifacts/video_runtime_v1/fast_attention_baseline_v1/REPORT.zh.md)，[模块距离表](../../artifacts/video_runtime_v1/fast_attention_baseline_v1/MODULE_DISTANCE.zh.md)，命令见local_setup/CURRENT_VIDEO_BASELINE.txt。采用旧Page64直接LUT精算＋完整128维V均值/rank8补偿与LSE合并；保留当前Risk、摘要、Full_Pset及A-only/split Reader，RoPE仍为逐位等价旧融合核。PAGE32、lazy关闭，旧Risk/旧摘要未启用。VideoRuntime默认legacy_components=None解析为attention；benchmark_hotspots默认attention；显式字符串覆盖默认以保留旧控制复现。完整当前配置为--reader-layout --reader-split --legacy-components rope,attention。GPU3同卡9chunks/105帧：生成12.749→11.325秒（1.126×），含VAE17.061→15.626秒（1.092×）；Dense12.812秒，当前/Dense加速比0.820×。切换误差PSNR17.330、SSIM0.6839、LPIPS0.2333、RGB MAE13.434/255已记录，不以数值不等价阻止用户明确指定的基线切换。新轨迹路由捕获位于artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt，H12/T72/Q64/P432，捕获运行latent与正式current逐位一致；后续热点实验优先使用此输入。

最新Reader向量阶段筛查已结束：[24组模块筛查](../../artifacts/video_runtime_v1/reader_vector_v1/REPORT.zh.md)、[单视频验收](../../artifacts/video_runtime_v1/reader_page32_v1/REPORT.zh.md)、[更新模块距离表](../../artifacts/video_runtime_v1/reader_page32_v1/MODULE_DISTANCE.zh.md)。向量维拆分、寄存器上限、CO打包无收益；仅将原有FP64→FP32页面转换提前共享有评分1.049–1.067×收益。完整路由双方交替预热5.309→5.073ms；首次5.290→6.145ms异常原始记录另保留。GPU3同卡9chunks短视频生成12.737→12.569秒（1.013×），含VAE17.036→16.884秒（1.009×），但latent NMSE0.23017、PSNR18.685、SSIM0.6942、LPIPS0.1950、RGB MAE11.669/255。固定输入诊断证明候选端点和log质量逐位一致，仅state归约变了（max9.54e-7）；闭环动作首分歧未逐事件定位。候选不推广，PAGE32及其他实验选项默认关闭，lazy也关闭。未来应保持归约布局来减少系数读取，或优化分母/动作评分融合。

最新热点优化与模块距离表：[按需回退短视频验收](../../artifacts/video_runtime_v1/lazy_video_v1/REPORT.zh.md)，[模块用时、参考理论下限与优化空间](../../artifacts/video_runtime_v1/lazy_video_v1/MODULE_DISTANCE.zh.md)。Reader两阶段先通过短视频latent/RGB逐位验收（生成12.984→12.738秒）；进一步lazy fallback移除常规路径两次FP64排除向量BMM（0.566 GFLOPs），整条路由5.288→4.704ms，短视频生成12.750→12.236秒，含VAE17.055→16.548秒。但lazy短视频latent NMSE0.022614、RGB MAE2.053/255，PSNR28.825、SSIM0.9559、LPIPS0.03172，并非逐位等价，保持可选关闭。数学表达式保留，回退求和次序改变；完整闭环首个动作差异尚未定位。只测试9 chunks/105帧、768×512、360历史页；所有模块实验独占GPU，分配0–7卡并行。模块距离使用本机大GEMM/复制参考值，明确是部分工作量下限，不能把差距当可承诺收益。下一热点是Reader向量阶段1.446ms和Risk约0.875ms（含softmax/U投影），其次分母及动作评分融合。

Reader两阶段详细记录：[验收报告](../../artifacts/video_runtime_v1/reader_split_v1/REPORT.zh.md)。可通过--reader-layout --reader-split --legacy-components rope启用；--lazy-scores仅启用有视频差异的候选。默认基线未切换。

最新逐位等价优化：[Reader A-only系数布局验收](../../artifacts/video_runtime_v1/reader_layout_v1/REPORT.zh.md)。216/288/360/432页完整评分、动作、no-op均逐位一致；432页Reader＋评分约1.28×，整条路由6.783→5.663ms。GPU3同卡9 chunks短视频生成14.806→13.993秒（1.058×），含VAE18.305秒，latent/RGB逐位一致。再叠加逐位等价RoPE为12.993秒、含VAE17.310秒，仍逐位一致。通过`--reader-layout`启用A-only，可加`--legacy-components rope`；默认开关尚未改动。CO布局实验有评分/动作差异且无收益，未采用。优化后Reader仍约2.29ms/40%路由时间、255寄存器，继续针对其标量/向量阶段的寄存器压力做profile和隔离优化。

最新实现复用实验：[旧版优化实现接回当前Full_Pset运行时](../../artifacts/video_runtime_v1/legacy_reuse/REPORT.zh.md)。只跑9 chunks/105帧短视频。当前所有权重与Full_Pset不变，复用旧RoPE、Risk、Page64精算＋完整均值/rank8补偿、摘要投影。GPU5同卡：生成14.770→11.902秒（1.241×），含VAE19.058→16.192秒（1.177×），仍慢于Dense 13.046秒。组合latent NMSE0.203949、PSNR19.324、SSIM0.7051、LPIPS0.1745；只换RoPE可逐位等价且生成13.796秒。候选使用`--legacy-components rope,risk,attention,summary`显式启用，默认基线未改变。模块profiling及理论工作量、短视频误差均留档。

最新消融已完成：[摘要权重 × Full_Pset 四组消融](../../artifacts/weight_module_ablation_20261117/REPORT.zh.md)，[用户指定格式表格](../../artifacts/weight_module_ablation_20261117/TABLE.zh.md)。按用户追加授权使用全部8卡，两个原30秒场景；四组均为第八轮运行时，只切换摘要权重和Full_Pset。旧/新摘要+Risk平均含VAE为74.05/73.97秒，前段LPIPS为0.3354/0.3341、后段0.6046/0.5537；旧/新摘要+Full_Pset为99.36/99.43秒，前段0.3673/0.3551、后段0.5828/0.5581。新摘要后段改善；Full_Pset在新摘要上增加25.45秒（约34%），中段LPIPS改善，前后段均值无改善。权重核对仅pool_k/pool_v/pool_feature变更，Risk/分组/rank8基不变。所有重复视频逐位一致，无非有限评分或同卡竞争。

最新复测：按用户明确要求恢复原记录的两个 30 秒场景，768×512 / 41 chunks / 5 历史 chunks / seed101，GPU6–7 上运行已接受第八轮基线。结果见 [原配置对比报告](../../artifacts/comparison_current_baseline_20261117/REPORT.zh.md) 和 [完整表格](../../artifacts/comparison_current_baseline_20261117/summary.csv)。Dense 与旧记录 latent 逐位一致；当前 Full_Pset 含 VAE 为 99.73 / 99.68 秒，相对同卡 Dense 为 0.656× / 0.653×。旧 summary 版本不含 Full_Pset 且权重不同。此项仅复测，未恢复暂停的优化任务；常规验收仍仅短视频。

最新：[第九轮Risk融合与形状编译验收](evidence/video_runtime_v1/HOTSPOT_ROUND9.zh.md)。短视频生成14.414秒；对第八轮新增latent NMSE0.012925已留档，未默认替换已接受基线。当前新目标继续profile与优化验收，仅短视频。

最新单项修改：[补偿IEEE FP32数学等价实现](evidence/video_runtime_v1/IEEE_COMPENSATION.zh.md)。用户随后明确接受此前较大误差的近似组合，当前基线恢复BF16四乘积mode4；IEEE mode1保留作对照。[第八轮](evidence/video_runtime_v1/HOTSPOT_ROUND8.zh.md)已完成：动作合批优化，短视频和15秒均与基线latent/RGB逐位一致。以后仅做短视频验收，只有用户明确要求质量测试才跑15秒。

最新运行时优化收尾见 [理论加速与误差记录](evidence/video_runtime_v1/ROUND_CLOSURE.zh.md)。此前一轮按用户要求结束后，用户曾重新激活1.2× Dense目标；[第七轮](evidence/video_runtime_v1/HOTSPOT_ROUND7.zh.md)及[全部/保守优化视频对照](evidence/video_runtime_v1/OPTIMIZATION_VIDEO_COMPARISON.zh.md)已完成。第七轮曾暂停；用户随后接受近似基线并启动第八轮，目标尚未达到。

本文汇总截至E49的项目情况，供新会话了解已有代码与实验。内容依据此前会话记录及已生成报告整理。

## 项目在做什么

streaming_sparse研究流式视频生成中的混合稀疏Attention。KV按Page64组织，一部分页面执行精确Attention，其余页面使用摘要近似，两部分采用共同Softmax归一化。

研究中使用Risk选择初始精算集合，再尝试利用摘要预测一次页面交换的收益。实际目标是在固定精算预算下，降低真实Attention输出误差。现有实验既涉及摘要与路由，也涉及Reader训练、集合搜索、缓存与Kernel实现。

## 仓库和运行环境

- 本地仓库：`/home/shuchenweng/jzk/streaming_sparse`
- GitHub：`https://github.com/jurky123/streaming_sparse`
- 最近确认推送的提交：`6b52faa`，内容为E49数值审计。
- 此前关键提交：`aefb74f`（E46/E47）、`cfae4fc`和`c0c2718`（E48）。

此前实验使用的Python：

```text
/tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python
```

E49记录的环境是Python 3.10.12、PyTorch 2.6.0+CUDA 12.4、NumPy 2.2.6、mpmath 1.3.0、MPFR 4.1.0，机器架构为x86_64。此前可使用8张GPU。系统Python与实验虚拟环境的依赖不同。

工作区中有其他未提交修改和未跟踪文件，并非所有本地内容都属于最后一轮实验。原始数据、部分模型与大缓存主要位于`artifacts/`，Git中保存了代码、报告、小型结果表及部分完整审阅包。

## 当前研究配置

| 项目 | 实验中使用的配置 |
|---|---|
| KV页面 | Page64 |
| 页面摘要 | R4，四个Key代表 |
| Value摘要 | 128维均值＋rank8 detail |
| Query tile | Q64 |
| 路由及交换评分Query | M16，16个probes |
| 精算预算 | 约10%页面 |
| own页面 | 保留在精算集合中 |
| 初选 | Risk |
| Medium候选池 | 16个移出候选×32个加入候选，共512动作 |
| 修正 | 一次交换或no-op |
| 真实质量评价 | Q64、完整128维输出NMSE |

E29以后大部分Reader实验使用冻结的E24 Joint-C摘要、E27 Adaptive φ和训练集拟合后冻结的B16输出投影基。

## 摘要和Reader的实现

摘要包含四个Key代表、Value均值、低秩detail和统计量。历史与当前页面有对应的摘要构建路径。Key pooling中使用`2*tanh`限制pooling logits，随后将四个Key代表的均值重中心化到页面Key均值。

页面Reader是一个`712 → 64 → 85`的MLP，约51,157参数。页面特征由Key代表、Value均值、detail和stats分别归一化后拼接。输出85个数，组织成5组，每组16维响应系数和1个质量系数。

Query读出使用常数项与四维Adaptive φ，形成Query相关的局部响应预测`r`和质量修正预测`beta`。模型随后结合已有摘要页面质量及全局摘要中心，构造集合评分。

其局部响应分解为：

\[
 t=m[e^\beta r+(e^\beta-1)c],\qquad u=m(e^\beta-1),
\]

其中`c`是页面摘要输出与全局摘要输出之差在B16中的投影。实际稳定实现以正质量混合形式计算输出，避免直接构造不稳定的指数差和共同分母。

经过纯setloss训练后，`r`和`beta`的物理回归误差可能变大，但动作收益可以改善，因此这些输出在该分支中更接近结构化决策代理。

## 主要代码位置

| 路径 | 内容 |
|---|---|
| `local_experiments/review_v1/ops.py` | Summary、摘要构建、Risk特征与选择、部署精度处理 |
| `local_experiments/e24_joint_representatives/core.py` | 页面贡献、混合输出、真实NMSE、交换gain、路由 |
| `local_experiments/e29_local_reader/core.py` | LocalReader、局部响应目标、原联合训练目标 |
| `local_experiments/e30_beta_diagnostics/run.py` | `local_scores`，稳定Set评分及同信息Independent |
| `local_experiments/e42_local_coverage/core.py` | 缓存数据加载、原训练损失、冻结摘要复用 |
| `local_experiments/e43_fit/` | 小集Page Encoder与自由页面系数表对照 |
| `local_experiments/e44_fit_diagnostics/` | 优化器时钟、损失消融、Layer29教师诊断 |
| `local_experiments/e45_frozen_replay/` | P→L移植、跨状态偏好复用、邻近状态留出 |
| `local_experiments/e46_supervision_bridge/` | M16监督对齐、完整事件桥接、按层Reader |
| `local_experiments/e47_conditional_value/` | P_set冻结回放、真实条件价值分母 |
| `local_experiments/e48_tail_state/` | 尾部物理检查、pooling反事实、完整分布纯setloss训练 |
| `local_experiments/e49_numeric/` | FP64路径、MPFR参考、外向区间数值审计 |

这些路径主要是研究和验证实现。仓库还存在其他运行时、性能分析和Kernel相关代码，交接文件没有对全部工程路径重新做清单核验。

## 到目前为止的实验认识

E11–E23主要探索集合优化空间、摘要误差信息、响应方向、二阶矩和主动探测。真实集合优化有收益，但低成本误差预测的稳定性和成本一直存在困难。

E24联合训练四代表，改善了初选混合Attention质量；后续单步交换没有同步获得可靠收益。E25–E28逐步分离低维容量、Query函数、页面系数预测和全局中心化问题，形成了页面局部响应＋解析全局中心的结构。

E29–E42探索局部响应Reader、β预测、共享校准、Key探测、矩近似、训练损失及监督覆盖。多个教师对照说明真实信息有价值，但多数在线预测方案未形成一致可靠改善。

E43–E47确认了小Reader的有限决策可拟合性，同时显示小集能力的适用范围、物理回归和动作目标的差异，以及近邻状态协议中真实条件价值很小。

E48补齐完整分布纯setloss训练，获得了新的跨种子开发改善。E49核查极小正分是否由数值误触发，结果不支持用浮点底噪解释两个灾难动作。

## 最近几轮的关键结果

### 小集可拟合性与损失消融

E43采用48个训练事件，每事件两个集合状态，共96状态，训练2304步。P为原Page Encoder，L为自由页面系数表。

| 模型 | Seed17 Oracle恢复 | Seed29 Oracle恢复 |
|---|---:|---:|
| 初始E29 | 18.41% | 34.52% |
| E43联合训练P | 37.86% | 64.43% |
| E44纯setloss P | 61.21% | 67.99% |

纯setloss提高了小集决策收益，同时未加权局部响应MSE升至约1.02，β误差也没有同步改善。

E45将P最终系数移植到L，两个种子各144状态的响应、β和动作评分逐位一致。L可以表示P找到的配置；原L较差不能解释为其函数表达空间更小。

### 监督对齐和完整事件桥接

E46把小集setloss标签改为M16真实收益后，Q64评价恢复率为34.90%／50.95%，低于原Q64监督的37.86%／64.43%。按层解耦结果不一致，没有同时改善完整训练和开发可靠性。

小集训练的P_set冻结回放到完整训练Risk事件，恢复率只有2.81%／10.31%；开发为17.26%／23.30%。这与原小集高恢复是不同范围的结果。

### 近邻状态条件价值

E47计算“各状态分别选择最佳动作”相对“固定动作但允许完美接受判断”的收益差。

- 两个已训练状态：条件增量占Oracle收益0.081%／0.011%。
- 三个近邻状态：1.025%／0.255%。

因此，那组近邻协议主要检验稳定性，不能有力证明或否定更广范围的集合条件学习。

### E48完整分布纯setloss

从E29 selected出发，按原E39状态计划做2304次更新，仅优化setloss；两个种子均完成完整训练／开发冻结评价。

| 开发集指标 | Seed17 | Seed29 |
|---|---:|---:|
| Set Oracle恢复 | 26.00% | 28.85% |
| 汇总Attention NMSE改善 | 3.603% | 3.693% |
| H_loss | 0.035994 | 0.054685 |
| 退化事件 | 69/288 | 73/288 |

这些是局部Attention输出质量，不是最终视频质量或端到端加速数据。净收益获得了进展，但可靠性与尾部仍未解决。

### 两个尾部事件

```text
data/forest_run_s702/c8_l29_s0_h9_t39.pt
data/reef_fish_s703/c5_l29_s0_h9_t37.pt
```

对应灾难情形中，重要页已经被Risk精算保护，错误交换将其移出。某些Query的真实页面质量接近1，页面内部token响应极尖锐。

E48单页Key的tanh／重中心化2×2反事实没有稳定消除灾难。两个关键移出页的原基线逐位重建，其他端点存在BF16路径差异。该实验冻结了Value、stats和原路由，不等于完整新摘要构建器的训练或部署结果。

### E49数值审计

冻结E29、匹配预算T0和E48 Full_Pset，比较历史FP64、统一归约FP64、交换增量、外向舍入区间及少量高精度参考。

- 全部8640个模型事件中，三条FP64路径选中同一最高动作。
- 区间规则未拒绝任何历史正提议。
- 实际收益和损害均未改变。
- 两个旧E29灾难动作的raw预测收益，高精度下仍为正，约为4.92e−11和1.05e−13。

旧归约路径对同一mask存在可测差异，但没有改变本次实际最高动作。现有证据将这些灾难归为可分辨的模型误判，而不是已证实的浮点底噪误触发。区间保证的范围是固定输入、指定算术表达式和记录的平台条件，并非真实收益安全保证。

## 数据和缓存

主要artifact目录：

```text
artifacts/e24_joint_representatives_v1/
artifacts/e29_local_reader_v1/
artifacts/e39_action_margin_v1/
artifacts/e40_reader_scale_v1/
artifacts/e48_tail_state_v1/
artifacts/e49_numeric_v1/
```

规范事件通常包含：

```text
q, N, Z, dense, page_logz, page_output, log_scale,
own, trajectory, layer, head, chunk, step, tile
```

固定QKV时，真实页面贡献可以缓存复用。E40另保存冻结摘要，后续多轮实验主要复用这些摘要和规范贡献。数据划分为8条训练轨迹、2条开发轨迹、4条Test轨迹；截至E49，Test仍未评价。

## 文档和完整结果位置

当前报告与入口：

- `docs/research/E46_E47_READING_ORDER.zh.md`
- `docs/research/E49_READING_ORDER.zh.md`
- `docs/research/evidence/e48_tail_state_v1/REPORT.md`
- `docs/research/evidence/e49_numeric_v1/REPORT.md`

历史总览：

- `docs/research/E11_E29_EXPLORATION_ROADMAP.zh.md`
- `docs/research/E11_E37_FINDINGS_AND_NEXT_STEPS.zh.md`

E46、E47、E48的完整审阅包已放入对应仓库证据目录。E49完整评分分为8个包，另有高精度输入／源码包，文件名、体积和SHA在：

```text
docs/research/evidence/e49_numeric_v1/packages.json
```

审阅包包含完整评分及部分固定模型／参考输入，但仓库并不包含全部原始训练数据；仅靠报告和代码不能重新生成所有历史实验。

## 当前阶段概况

当前已有低成本结构化评分器、完整分布纯集合训练带来的有限收益，以及较完整的误差与数值诊断。尚未形成经过端到端质量、性能和可靠性统一验证的部署方案。

此前已收束固定摘要Reader上的零散校准与参数搜索，保留E48 Full_Pset作为有实际改进的研究基线。新的摘要／初选分支尚未在这些最近实验中训练。工程性能实现与研究原型并存，二者的覆盖范围和成熟度需要从具体运行入口判断。
