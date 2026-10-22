# Streaming KV Sparse Routing v0.3

按项目根目录的 `Streaming_KV_Sparse_v03_Experiment_Implementation_Guide (2).md`
实现。v0.2 与公共底层源码保持原样；v0.3 只导入并复用底层实现。

当前只完成软件实现及合成数据测试，未开始真实数据实验。
旧 QKV tensor 尚未恢复，已有 JSONL 索引不能代替数据。
Backbone 和 Wan 模型资产已备齐，但不能替代 QKV。

## 架构与约定

- 唯一 Joint Feature：`[K;V] -> 256 -> 256 -> 128 -> 128`；中间 GELU。
  同一 Feature 同时进入原 `soft_kmeans` 和聚合路径，不输入 position。
- 聚合：`Linear(128,128)+GELU` 变换，再 Mean/Max/Min，`384 -> 128` 投影。
- Pre-Q：一层 128 hidden / 4 heads / FFN512 的 pre-norm Transformer。
- Q：保留 tile 内 mean pooling 和两层 MLP 形式，使用 `128 -> 256 -> 128`。
- A：`Linear([h,q,h*q], 384 -> 128)+GELU`。
  B：`Linear(q, 128 -> 256)` 生成 gamma/beta，`gamma*h+beta`。
- Post-Q：两层与 Pre-Q 同规格的 Transformer，最后一个 Linear 输出 score。
- Transformer 显式实现 QKV/输出/FFN 投影以支持 pair 参数残差；dropout=0，
  不使用序列位置或 block index 编码。Pooling/聚类/选择不添加可学习参数。
- 所有 Linear 使用 `W0 + U[layer,head] @ V[layer,head]`，rank=1，bias 共享；
  LayerNorm affine 参数也有 pair 残差。pair 直接枚举三层 × 十二头，
  不分解为 layer residual + head residual。两因子以小非零值初始化以接收首步梯度。
- 两种 Fusion 共用全部其他设置。Concat：891,137 shared + 396,324 residual；
  FiLM：874,881 shared + 391,716 residual。禁止 residual 总容量达到 shared 容量。
- 只计算 normalized attention output SSE；聚类 usage regularizer 仅作为诊断值，
  不加到 loss。门控复用原 hard-forward / soft-backward token budget 实现。
- 最终 attention 使用原始 K/V，全部 current KV 保留且不计入历史预算。
- AdamW、lr=1e-4、weight_decay=.01、grad_clip=1 延用原训练默认设置。
  未由指引指定的细节已写入 `configs/protocol.json`，不做额外超参搜索。

## 本轮文件

`model.py` 模型；`data.py` 数据协议与缺失检查；`train.py` 训练/恢复；
`evaluate.py` B0/B1/B2/B3/B4 或 B5；`metrics.py` 诊断；
`sanity.py` 单 snapshot 正确性；`gate.py` 4-head 架构门槛；
`capture.py` 缺失八个 heads 的采集计划；`merge.py` 内容寻址合并；
`workflow.py` 只打印命令矩阵；`analyze.py` 跨 seed 汇总。

## 软件验证

从 Self-Forcing 根目录激活 `streaming_sparse/.venv` 后：

```bash
python -m unittest streaming_sparse.v03.test_correctness -v
python -m streaming_sparse.v03.sanity --synthetic --device cpu \
  --output streaming_sparse/v03/reports/synthetic_correctness_cpu.json
```

合成测试包括少量临时 optimizer 更新以验证训练恢复；数据和 runs 均在临时目录。
`sanity` 仅 forward/backward，optimizer_updates=0。
CPU 和 A100 的合成结果均不能通过真实数据实验门槛，不能作为研究结论。

## 取得真实 QKV 后的执行顺序

以下是未来执行命令，本次未运行。`TRAIN_CACHE_VERSION` 替换为一个 train snapshot ID。
阶段间不自动推进；任一步未通过时保留结果并停止扩大规模。

```bash
DATA_V03=streaming_sparse/v02/data/full
python -m streaming_sparse.v03.data --data "$DATA_V03"
python -m streaming_sparse.v03.sanity --data "$DATA_V03" --device cuda \
  --output streaming_sparse/v03/reports/real_correctness.json
```

接着对 A/B 各做一次固定 snapshot、四个 step、固定首 tile、keep=.2 的 overfit。
`overfit_*` 配置使用 200 updates 作为实现门槛预算；最终 hard error 必须低于初始值。
该阶段生成 `overfit_gate.json`，不声称泛化能力。

```bash
python -m streaming_sparse.v03.train --config streaming_sparse/v03/configs/overfit_concat.json \
  --data "$DATA_V03" --overfit-snapshot TRAIN_CACHE_VERSION \
  --output streaming_sparse/v03/runs/overfit/concat
python -m streaming_sparse.v03.train --config streaming_sparse/v03/configs/overfit_film.json \
  --data "$DATA_V03" --overfit-snapshot TRAIN_CACHE_VERSION \
  --output streaming_sparse/v03/runs/overfit/film
```

打印 4-head pilot 命令：各 Fusion seed0，1500 updates；只打印，不执行。

```bash
python -m streaming_sparse.v03.workflow --stage pilot --data "$DATA_V03"
```

正式训练要求前述真实正确性及对应 Fusion 的 overfit gate 通过。
每 500 updates，用固定 validation Prompt × Layer × Head 子集的全部四个 step、
全部保存 tiles 验证；**只以 keep=.2 的 macro prompt-head relative error 选 best.pt**。
Validation subset 固定取 seed0、最短 history，保留全部四个 validation prompts。
Checkpoint 含配置、optimizer、Python/Torch/CUDA RNG、数据/源码/包身份和诊断。
诊断取最近一次 optimizer 更新前的 training forward，checkpoint 明确记录该时序。
显式 `--resume <原 run>/latest.pt` 才恢复；中断在 validation 内时先恢复该 journal。
没有 Supervisor 或自动启动脚本。

Pilot 的 best.pt 需在完整 Val 上评估 .3/.2/.1，命令由 workflow 给出。
B3 从同一 checkpoint 的 Feature 得到与 B4/B5 完全相同的 partition，分别计算 Mass；
**不得把 A 的 B3 混用到 B**。另记 learned 实际 token cost 下的 Mass 误差。
B0 Spatial+Mass、B1 Raw-K+Mass、B2 Raw-KV+Mass 使用原有 baseline 实现。
可显式 `evaluate --oracle` 做慢速 greedy 诊断，不参与训练，也不是最优上界。

```bash
python -m streaming_sparse.v03.gate \
  --summaries streaming_sparse/v03/runs/pilot/concat/seed0/evaluation_val/summary.json \
              streaming_sparse/v03/runs/pilot/film/seed0/evaluation_val/summary.json \
  --training-runs streaming_sparse/v03/runs/pilot/concat/seed0 \
                  streaming_sparse/v03/runs/pilot/film/seed0 \
  --overfit-reports streaming_sparse/v03/runs/overfit/concat/overfit_gate.json \
                    streaming_sparse/v03/runs/overfit/film/overfit_gate.json
```

实现采用保守的扩大规模规则：两种 Fusion 在主预算均不劣于同 partition Mass，
且至少一种更好；同时要求完整 288 个 validation snapshots、1500-update pilot、
有效梯度与真实 overfit 证据。此规则是执行门槛，不是统计显著性或成功结论。
诊断保留 soft-hard gap、score/Mass 相关性、step-wise mask Jaccard、feature norm、
block size/entropy/repair，供人工分析；未训练的合成模型 mask 不变并不代表代码错误。

## 后续 12-head 阶段

先打印 supplemental capture 命令：仅 heads 1/2/4/5/7/8/10/11；无 `--execute` 不采集。
只有 current-source `architecture_gate.json` 通过才允许显式启动。

```bash
python -m streaming_sparse.v03.capture --output streaming_sparse/v03/data/extra_heads
```

在明确决定采集后才加 `--execute`。采集使用原 v0.2 分层 sampler、冻结的 24 prompts、
seeds0/1、layers4/14/24、frames12、tile16、8 tiles/event、SDPA，写入新目录。
已有四个 heads 不重采、不覆盖。合并也要求新目录：

```bash
python -m streaming_sparse.v03.merge --sources "$DATA_V03" \
  streaming_sparse/v03/data/extra_heads --output streaming_sparse/v03/data/full
python -m streaming_sparse.v03.data --data streaming_sparse/v03/data/full \
  --heads 0 1 2 3 4 5 6 7 8 9 10 11
python -m streaming_sparse.v03.workflow --stage full --data streaming_sparse/v03/data/full
```

Merge 在写入前检查相同 `(prompt,seed,layer,head,history,step)` 的冲突，
仅接受 tensor 内容完全一致的重复单元；目标文件以内容摘要命名。
目标完整覆盖 20,736 events / 5,184 snapshots，沿用原 split。
Full 矩阵为 A/B × seeds0/1/2、每个 3000 updates；Val 选择 checkpoint 后才评估 Test。
`workflow --shared` 只生成 Shared ablation 命令，应在 A2 通过后再考虑。

最终可用 `analyze --reports <同一组各 seed 的 summary.json> --require-full-seeds \
--output <结果.json>` 汇总；要求完整 12 heads、seeds0/1/2、相同源码和数据。
报告 macro prompt-head、global SSE/energy、prompt 内 energy pooling 后宏平均、tile 分位数、
actual keep、budget utilization、逐 seed 同 partition learned/Mass 比值及 seed 波动。
全部真实实验结果当前均未产生。
