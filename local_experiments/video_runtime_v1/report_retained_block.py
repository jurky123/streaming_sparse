"""Summarize the bounded complete-block capture experiment."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/video_runtime_v1/block_retained_v1'

def run():
 rows=[]
 for tier in ('base','intersection','step'):
  a,b=[json.loads((OUT/f'video/{tier}_{v}/result.json').read_text()) for v in ('old','new')]
  za,zb=[torch.load(OUT/f'video/{tier}_{v}/latents.pt',weights_only=True) for v in ('old','new')]
  sa,sb=a['runtime_stats'],b['runtime_stats']
  row=dict(tier=tier,before_s=a['median_seconds'],after_s=b['median_seconds'],speedup=a['median_seconds']/b['median_seconds'],saved_ms=(a['median_seconds']-b['median_seconds'])*1000,latent_bitwise=torch.equal(za,zb),latent_nmse=((za.float()-zb.float()).square().sum()/za.float().square().sum()).item(),before_cold_s=a['cold_seconds'],after_cold_s=b['cold_seconds'],runtime_stats_before=sa,runtime_stats_after=sb)
  row['warm_videos_to_amortize_extra_cold']=(row['after_cold_s']-row['before_cold_s'])/(row['before_s']-row['after_s']) if row['before_s']>row['after_s'] else None
  rows.append(row)
 modules={t:json.loads((OUT/f'{t}.json').read_text()) for t in ('base','intersection','step','mixed_scope_fixed')}
 result=dict(rows=rows,module_contracts=modules,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'local_experiments/video_runtime_v1').glob('*.py')})
 (OUT/'summary.json').write_text(json.dumps(result,indent=2))
 table='\n'.join(f"| {r['tier']} | {r['before_s']:.6f} | {r['after_s']:.6f} | {r['speedup']:.4f}× | {r['saved_ms']:.2f} | {r['latent_bitwise']} |" for r in rows)
 cold_table='\n'.join(f"| {r['tier']} | {r['before_cold_s']:.3f} | {r['after_cold_s']:.3f} | {r['warm_videos_to_amortize_extra_cold']:.0f} |" for r in rows)
 module_table='\n'.join(f"| {tier} | {r['step']} | {r['eager_ms']:.4f} | {r['graph_ms']:.4f} | {r['speedup']:.4f}× |" for tier,d in modules.items() if tier!='mixed_scope_fixed' for r in d['rows'] if 'speedup' in r)
 (OUT/'MODULE_DISTANCE.zh.md').write_text(f'''# 当前三档整块 CUDA Graph 有限实验

## 改动与范围

保留当前 split summary、稳定 spatial softmax、精算和补偿算子、M16 路由以及三档 Full_Pset 策略。只将 layer 1–29、chunk 2–8 的 noisy transformer block 捕获；clean commit 和前两 chunk 仍为原执行方式。

- `local_experiments/video_runtime_v1/block_graph.py`：不再强制 `inline_summary=True`，缓存键纳入 sparse、实际修正分支与 head 张量身份；图持有 head 张量防止地址复用。固定 cross-attention K/V 缓冲，在源张量身份改变时复制，以支持 reset 后重新分配的文本 K/V。
- `runtime.py::correction_scope`：捕获与 eager 使用同一分支条件；clean 页编码仍由层启用状态控制，与当前 noisy step 是否修正分开。
- `runtime.py::forward`：捕获期间继续使用已部署 split summary，不切换到 `SummaryOps.build`。
- `accepted_tiers.py --block-graph`：显式实验入口，默认关闭。
- `check_retained_block.py`：真实 chunk8/layer5/P432 同输入比较与交替计时；`--mixed-scope` 检查同一进程的分支切换。

## 模块验收

同一场景 city_walk/seed701，完整历史来自当前部署 latent 的 clean commit。四个 noisy step 的 block 输出、重复调用输出、GPU 逻辑计数均逐位一致。额外检查关闭、head子集 [4,3,9]、全head、另一子集 [1,2,7]、再次关闭，输出均逐位一致；另测 cross-attention K/V 重新分配以及 V 内容改变后重放。

计时完成预热，20轮交换先后顺序，每次5调用；表中为包含输入/输出复制的同步墙钟中位数。

| 档位 | step | 原 block/ms | 整块图/ms | 加速比 |
|---|---:|---:|---:|---:|
{module_table}

## 短视频验收

9 chunks，105帧，768×512，16 FPS，history360，tile72；city_walk/seed701。每档同一GPU先原路径后整块图，完整视频预热后各5次。仅生成，不含VAE、编码、IO。三档分配独立GPU。速度以本轮原路径为分母，不与先前不同轮次绝对耗时直接相比。

| 档位 | 原生成/s | 整块图生成/s | 加速比 | 节省/ms | latent逐位一致 |
|---|---:|---:|---:|---:|---|
{table}

## 过程中修复的跨视频问题

初版单 block 与混合分支检查通过，但全视频预热后第二遍出现非有限值。定位到 `wan/modules/model.py::WanT2VCrossAttention.forward` 在 `is_init=False` 时替换 cache 内的 K/V 张量；捕获图保留旧地址，原字典更新不改变图中的指针。修复为图使用固定 K/V 缓冲，新视频源 K/V 改变时更新。失败日志与代码快照保存在 `*_new_failed_cross_cache`，不是已验收实现。

## 与理论距离的关系

此前当前基线 chunk8 timeline 区间955.909ms，GPU区间并集926.642ms，空隙29.267ms，占3.06%。这个数字包含profile影响，不是全部可回收的Python时间。整块图不减少模型FLOPs；它减少CPU发射和部分中间图调度，同时新增x/context/e复制和block输出复制。因此验收看视频净收益。

全部层捕获的冷启动明显重于模块测试。下表使用本轮记录，按额外冷启动时间除以单视频稳定节省时间，估算摊平所需的后续视频数；模型加载不包含在这段冷运行计时内。

| 档位 | 原冷运行/s | 整块图冷运行/s | 摊平额外冷启动需后续视频数 |
|---|---:|---:|---:|
{cold_table}

保持 `--block-graph` 显式选择，**不修改部署默认值**。当前20视频量级评估不值得支付额外捕获成本；长驻同形状且重复数足够多的进程可以使用本原型。稳态生成收益与包括捕获的任务吞吐是两个口径。当前只评估固定短视频；没有运行15s、30s，也没有重跑算法质量评价。若latent一致，无须重新解码证明同一输入的确定性VAE输出。

## 复现

```bash
CUDA_VISIBLE_DEVICES=6 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.video_runtime_v1.accepted_tiers --tier base --reps 5 --skip-profile --local-assets /tmp/streaming_sparse-0/runtime --block-graph --output <fresh-dir>
```

原路径删除 `--block-graph`，其他两档替换 `--tier`。原始数据位于本目录 `video/*/result.json`、`latents.pt`、`source/` 与各档模块JSON。报告由 `report_retained_block.py` 生成。
''')
 print(json.dumps([{k:v for k,v in r.items() if not k.startswith('runtime_stats')} for r in rows],indent=2))
if __name__=='__main__':run()
