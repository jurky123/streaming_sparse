"""Record module work, short-video acceptance and implementation references."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/block_pointwise_v1'
def main():
 module=json.loads((OUT/'module.json').read_text());rows=[]
 for tier in ('base','intersection','step'):
  a,b=[json.loads((OUT/f'video/{tier}_{v}/result.json').read_text()) for v in ('old','new')]
  za,zb=[torch.load(OUT/f'video/{tier}_{v}/latents.pt',weights_only=True) for v in ('old','new')]
  rows.append(dict(tier=tier,before_s=a['median_seconds'],after_s=b['median_seconds'],speedup=a['median_seconds']/b['median_seconds'],saved_ms=1000*(a['median_seconds']-b['median_seconds']),latent_bitwise=torch.equal(za,zb),latent_nmse=((za.float()-zb.float()).square().sum()/za.float().square().sum()).item(),counters_bitwise=a['runtime_stats']['counters']==b['runtime_stats']['counters'],before_cold_s=a['cold_seconds'],after_cold_s=b['cold_seconds']))
 passed=module['passed'] and all(r['latent_bitwise'] and r['counters_bitwise'] for r in rows)
 assert passed
 summary=dict(rows=rows,module=module,passed=passed,kernels=json.loads((OUT/'kernel_metadata.json').read_text()),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'local_experiments/video_runtime_v1').glob('*.py')})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 table='\n'.join(f"| {r['tier']} | {r['before_s']:.6f} | {r['after_s']:.6f} | {r['speedup']:.4f}× | {r['saved_ms']:.2f} |" for r in rows)
 mt='\n'.join(f"| {r['name']} | {r['before_ms']:.6f} | {r['after_ms']:.6f} | {r['speedup']:.3f}× |" for r in module['rows'])
 (OUT/'MODULE_DISTANCE.zh.md').write_text(f'''# 主干调制与残差融合：模块归因、理论工作量及三档验收

## 结果与部署

四类逐元素操作融合已通过32组模块契约及三档9-chunk视频验收。三个档位的latent与逻辑计数逐位一致，没有引入新的数值近似。开启 `block_pointwise`，整块 CUDA Graph 仍默认关闭。层选择、head选择、step选择、权重、M16与512动作均不变。

| 档位 | 原生成/s | 融合后生成/s | 加速比 | 节省/ms |
|---|---:|---:|---:|---:|
{table}

每档同一GPU先原路径后新路径，完整视频预热后各5次取中位数；三档分别GPU6/7/5。city_walk/seed701，9chunks，105帧，768×512，16FPS，history360，tile72；仅生成，不含VAE/编码/IO。没有执行15s或30s。显式开关及完整源码快照保存在 `video/*/result.json` 与 `source/`，所以默认配置的后续更新不会改变本轮证据。

## 热点定位

真实满窗口一次 noisy step（30层）原路径中，mod1约2.154ms、res1约1.975ms、mod2约2.164ms、res2约1.888ms，合计8.181ms。`capture_pointwise.py` 在真实block内部添加四个标签；`profile.json` 同名CPU/GPU annotation是两个口径，不能重复相加。这里取CPU annotation归属的设备kernel时间。

原模块先写出乘法中间结果，随后再读它进行加法。新增kernel只改变这条数据流；LayerNorm、RMSNorm、线性层、FFN GELU、attention全部保持原实现。优化作用于30层的noisy和clean forward，与Full_Pset档位无关。

## 数学和舍入语义

以 `R` 表示写回BF16时的舍入：

- 调制：`R(R(x * R(1 + scale)) + shift)`。
- 残差：`R(x + R(y * gate))`。

kernel中显式执行每个 `.to(BF16).to(FP32)`，禁止FMA融合（`enable_fp_fusion=False`）。虽然不再物化中间数组，中间舍入仍存在。原模块使用的BF16权重、门控、输入和输出均未改精度。PTX保留BF16转换、无 `fma.rn`；两种kernel均26寄存器、0spill、0shared。

## 模块隔离计时

真实layer5/chunk8/step0，张量 `[1,4608,1536]`；分别capture原表达式与融合kernel，预热后30轮交替顺序，每轮20次重放。该表是CUDA Graph中的模块耗时，不能直接等同完整视频节省量。

| 操作 | 原模块/ms | 融合/ms | 加速比 |
|---|---:|---:|---:|
{mt}

契约包含四个真实模块×真实/随机/小/大四种数值尺度×连续/原始6倍frame stride两种布局，共32组，全部逐位一致。视频另比较最终latent与四项逻辑计数，也全部一致，因此没有额外解码质量实验。

## 优化后profile复核

`accepted_profile/detail.trace.json` 的实际kernel事件中，600次 `_pointwise` 合计11.386ms，覆盖一个chunk的30层×5次forward×4项操作。按原单step8.181ms和相同形状作5倍估算，原四项约40.905ms，差约29.52ms/chunk；9chunks约0.266s，与视频实测节省0.252–0.268s一致。这是独立profile间的近似归因，不是逐事件严格相减。统计直接取 `cat=kernel`，不重复累计CPU/GPU同名annotation。

## 理论读写量和剩余空间

单个大张量有7,077,888个BF16元素，占14.156MB。下面只计算必须访问的大数组，小型scale/shift/gate被多次逻辑读取但可被缓存，不能把这些重复读取全部算作HBM流量。

| 操作 | 原大数组读写/MB | 融合后/MB | 减少 |
|---|---:|---:|---:|
| 一次调制 | 56.623 | 28.312 | 50% |
| 一次残差 | 70.779 | 42.467 | 40% |
| 每block两次调制＋两次残差 | 254.804 | 141.558 | 44.44% |

以既有实测复制1266.396GB/s换算，新调制约0.0224ms、新残差约0.0335ms，是**流量参考而非硬件下限**；这些工作集可能部分命中40MB L2。原调制加速超过2×，不能仅用HBM字节减少解释：广播索引、kernel发射、临时分配与缓存行为均随融合改变。新调制还对每个元素重复计算 `1+scale`，算术工作并未全部减少，但避免中间物化的收益更大。

首版没有做warp/block组合筛查。归一化仍是独立操作，后续若融合必须单独验证其归约树和BF16输出；不能把本轮无归约改动的数值结论套用过去。

## 实现与复现

- `local_experiments/video_runtime_v1/block_pointwise.py::_pointwise`：两种融合kernel；`reference`为原表达式；`PointwiseBlocks`包装原block，保留其他子模块调用。
- `deployment.py::make_runtime`：统一接入benchmark及常驻worker；`CURRENT.block_pointwise`记录部署开关。
- `runtime.py::close`：恢复包装前的block forward。
- `accepted_tiers.py`：`--block-pointwise / --no-block-pointwise`配对复现。
- `capture_pointwise.py`：真实输入与热点归因；`check_pointwise.py`：数值契约、交替计时；`report_pointwise.py`：本报告。
- `profile_retained.py --block-pointwise`：融合后的全chunk profile，保存在 `accepted_profile/`。

```bash
CUDA_VISIBLE_DEVICES=6 /tmp/streaming_sparse-0/runtime/streaming_sparse/.venv/bin/python -m local_experiments.video_runtime_v1.accepted_tiers --tier base --block-pointwise --reps 5 --skip-profile --local-assets /tmp/streaming_sparse-0/runtime --output <fresh-dir>
```

对照使用 `--no-block-pointwise`，其余参数不变。本轮加速比相对同档当前实现，没有将Full_Pset策略缩减或算法相对Dense的误差计入优化收益。
''')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
