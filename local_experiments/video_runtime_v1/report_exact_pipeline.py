"""Bounded exact-attention lifetime trials, rounding and short-video differences."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/exact_pipeline_v1'
def main():
 module=json.loads((OUT/'module.json').read_text());windows=[json.loads((OUT/f'pages{p}.json').read_text()) for p in (216,288,360,432)];rows=[]
 for tier in ('base','intersection','step'):
  a,b=[json.loads((OUT/f'video/{tier}_{v}/result.json').read_text()) for v in ('old','new')]
  za,zb=[torch.load(OUT/f'video/{tier}_{v}/latents.pt',weights_only=True) for v in ('old','new')]
  quality=json.loads((OUT/f'video/{tier}_quality.json').read_text())
  rows.append(dict(tier=tier,before_s=a['median_seconds'],after_s=b['median_seconds'],speedup=a['median_seconds']/b['median_seconds'],saved_ms=1000*(a['median_seconds']-b['median_seconds']),latent_bitwise=torch.equal(za,zb),latent_nmse=((za.float()-zb.float()).square().sum()/za.float().square().sum()).item(),counters_before=a['runtime_stats']['counters'],counters_after=b['runtime_stats']['counters'],quality=quality))
 summary=dict(rows=rows,module=module,windows=windows,default_enabled=False,decision='Keep as explicit experiment: FP32 accumulation order changes; no new input approximation',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'local_experiments/video_runtime_v1').glob('*.py')})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 vt='\n'.join(f"| {r['tier']} | {r['before_s']:.6f} | {r['after_s']:.6f} | {r['speedup']:.4f}× | {r['latent_nmse']:.6g} |" for r in rows)
 qt='\n'.join(f"| {r['tier']} | {r['quality']['quality']['psnr']:.4f} | {r['quality']['quality']['ssim']:.6f} | {r['quality']['quality']['lpips']:.6f} | {r['quality']['rgb_mae_255']:.4f} | {r['quality']['rgb_rmse_255']:.4f} | {r['quality']['rgb_max_255']:.0f} |" for r in rows)
 mt='\n'.join(f"| {r['name']} | {r['ms']:.6f} | {r['speedup']:.3f}× | {r['registers']} | {r['nmse']:.3g} | {r['lse_bitwise']} |" for r in module['rows'])
 wt='\n'.join(f"| {r['pages']} | {r['count']} | {r['rows'][0]['ms']:.6f} | {r['rows'][1]['ms']:.6f} | {r['rows'][1]['speedup']:.3f}× | {r['rows'][1]['nmse']:.3g} |" for r in windows)
 (OUT/'MODULE_DISTANCE.zh.md').write_text(f'''# 精算AV运行累加器复用：收益与舍入轨迹

## 本轮结论

保留 `--exact-fold` 为显式实验，**默认仍关闭**。原型将高低项AV直接累加到已缩放的运行输出，模块加速约1.09–1.12×，但改变FP32加法顺序。没有降低输入/输出精度，没有减少高低项乘积，也没有改Page64/Q64、精算预算或Full_Pset策略。

本轮用同档短视频比较优化与原实现的差异；它不是相对Dense的质量结论。现有逐位一致部署基线保持。若输出轨迹不同，不能仅凭LPIPS相对旧实现非零认定相对Dense变差，也不能默认认为训练能消除这些轨迹差异。

## 先核对旧实验

`artifacts/comparison_20261113/integration_checks/kernel_optimization3/exact_launch_interleaved.json` 已测试stages2/3/4，stage2较快；`kernel_optimization4/exact_query_tuning.json` 已测试BQ16/32/64及4/8warps。此次不重复这些组合筛查。

初步源码假设“高低项dot共用累加器”被生成TTGIR否定：编译器已将第二个dot的acc设置为第一个dot结果。因此仅把源码写成 `dot(low,V,high)` 不会减少实际工作，未将它当成新优化。

## 硬件证据与新原型

真实P432、43精算页、layer15/chunk8/step0。原kernel 252寄存器、65536B shared、0spill；理论occupancy12.5%，实际12.44%。DRAM13.75%、SM计算49.93%、无eligible warp周期70.11%。不是饱和HBM主导。

新fold核237寄存器、shared仍65536B；实际occupancy12.45%，并未提升。SM计算54.65%，无eligible warp周期67.00%。因此收益不能说成“寄存器减少提高了occupancy”；它来自累加器组织及执行依赖的变化。NCU原始输出为 `ncu.csv`、`fold_ncu.csv`。

只试三种新组织：fold、延后volatile V加载、二者组合。V加载原型最终shared没有下降，且没有净收益，停止该方向。

| 原型 | ms | 加速比 | 寄存器 | attention NMSE | LSE逐位一致 |
|---|---:|---:|---:|---:|---|
{mt}

## 数值差异属于哪一种

基线编译结果先计算高项dot，再以其结果为acc计算低项dot，最后加上按alpha缩放后的运行状态。fold改为从运行状态开始依次执行高项、低项dot。

实数运算中的表达式相同，但FP32累加不满足结合律，Tensor Core内部的累加舍入位置也随acc起点变化。这里是**累加顺序导致的舍入差异**，不是新的低秩近似、预算缩减或BF16量化方案。QK、softmax、alpha、LSE路径不变，已测LSE逐位一致。

模块最大绝对误差约1.44e-4、NMSE约2.03e-10（P432）。微小模块差异可能通过BF16边界、后续路由及反复去噪放大，不能只用模块NMSE代替视频验收。

## 页面数补测

同一个真实输入裁取216/288/360页，LUT先保留合法页，再用未选页补齐10%预算，确保无重复页；P432为原真实LUT。其他窗口是合成前缀模块测试，不声称为对应chunk的真实生成输入。

| 页面数 | 精算页数 | 原/ms | fold/ms | 加速比 | NMSE |
|---|---:|---:|---:|---:|---:|
{wt}

## 短视频速度与实现差异

city_walk/seed701，9chunks，105帧，768×512，16FPS，history360/tile72。每档同一GPU先旧后新，完整预热后各5次中位数；仅生成，不含VAE、编码和IO。三档GPU6/7/5。没有运行15s或30s。

| 档位 | 原生成/s | fold生成/s | 加速比 | latent NMSE |
|---|---:|---:|---:|---:|
{vt}

PSNR/SSIM/LPIPS沿用RGB `[21,101)`，RGB差异覆盖全部105帧，范围0–255。对照双方均为同档、同prompt/seed。

| 档位 | PSNR | SSIM | LPIPS | RGB MAE/255 | RGB RMSE/255 | 最大差/255 |
|---|---:|---:|---:|---:|---:|---:|
{qt}

生成后的输出/LUT变化及接受计数记录在 `summary.json`。本轮只记录优化误差，不重跑Dense或训练诊断；不能把这些差异简单称为“可训练拿回的算法误差”。

## 闭环放大与基线核对

`trajectory.json` 证实本轮old三档latent与上轮已验收输出逐位一致。两边前两个dense chunk也完全一致；从第一个稀疏chunk2起，三档latent NMSE约0.00447/0.00262/0.00302，随后继续放大。整段latent NMSE约0.352/0.129/0.160。Full_Pset已评分事件数量不变，但接受次数由10531→10535、174166→174161，说明后续决策轨迹确实变化。

因此，收益只有约0.5–0.7%生成加速，不能凭局部2e-10 NMSE把该实现归入既有逐位一致优化。保留实验材料和关闭默认，不要求用户在不确定的误差上立即作决定。

## 工作量和剩余空间

当前精算主矩阵乘约116.87GFLOPs/层，基线约150TFLOP/s、fold约166TFLOP/s。三个主dot仍存在；仅省去少量独立FP32矩阵加法及改变累加器数据流，主FLOPs没有大幅减少。A100的寄存器/shared约束仍限制两CTA/SM，不能把剩余Tensor Core峰值差额当作可轻易收回的预算。

## 代码和复现

- `local_experiments/gap_exact_round_kernel.py::_attn_fwd`：未改动的部署基线。
- `local_experiments/video_runtime_v1/exact_fold.py`：显式运行acc的高低项AV。
- `exact_vlate.py`、`exact_fold_vlate.py`：已结束、不部署的加载组织原型。
- `profile_exact_ncu.py --fold`：硬件计数入口；`check_exact_pipeline.py`：四实现对照；`check_exact_fold_windows.py --pages`：窗口补测。
- `legacy_components.py::attention`、`deployment.py::configure`：显式开关，默认False。
- `accepted_tiers.py --exact-fold`：短视频实验；`quality_exact_fold.py`：同档短视频实现差异；`report_exact_pipeline.py`：本报告。

`video/*/source/`保存视频运行时源码，`latents.pt`及 `*_comparison.mp4` 保留并排复核材料。当前默认基线不包含fold。
''')
 print(json.dumps([{k:v for k,v in r.items() if k!='quality'} for r in rows],indent=2));print(qt)
if __name__=='__main__':main()
