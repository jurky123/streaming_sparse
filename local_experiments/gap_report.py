"""Final paired quality, wall-time and calibrated cost report; no hidden promotion."""
import json
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/four_gaps_20261031')
def main():
    rows=[];labels=['initial','old16','selected_shared','selected_matched']
    for seed in [404,505]:
        for folder in sorted((ROOT/f'test_s{seed}').glob('test_*')):
            metrics={k:json.loads((folder/(k+'.json')).read_text())['quality'] for k in labels if (folder/(k+'.json')).exists()}
            rows.append(dict(case=folder.name,metrics=metrics))
    complete=len(rows)==12 and all(len(r['metrics'])==4 for r in rows)
    means={k:{m:sum(r['metrics'][k][m] for r in rows if k in r['metrics'])/sum(k in r['metrics'] for r in rows) for m in ['lpips','ssim','psnr']} for k in labels}
    gates={}
    for candidate in labels[2:]:
        gates[candidate]={}
        for baseline in ['initial','old16']:
            pairs=[r for r in rows if candidate in r['metrics'] and baseline in r['metrics']]
            bad=[r['case'] for r in pairs if r['metrics'][candidate]['lpips']>r['metrics'][baseline]['lpips']+1e-6 or r['metrics'][candidate]['ssim']<r['metrics'][baseline]['ssim']-1e-6]
            gates[candidate][baseline]=dict(status='fail' if complete and bad else 'pass' if complete else 'pending',regressions=bad)
    timings={}
    for key,folder,pattern in [('reference','performance','reference_*.json'),('graph','performance','graph_*.json'),('fast','performance','fast_*.json'),('old16','performance','old16_*.json'),('dense','performance_dense','dense_*.json')]:
        values=[json.loads(p.read_text())['performance']['total_generation_vae_seconds'] for p in sorted((ROOT/folder).glob(pattern))]
        timings[key]=dict(seconds=values,mean_seconds=sum(values)/len(values) if values else None)
    result=dict(complete_quality=complete,means=means,gates=gates,rows=rows,performance=timings,quality_promoted=False,scope='Quality uses six previously seen test prompts with new seeds404/505; 12 seeded units, not12 independent scenes. Performance is matched_e1 horse_field seed101, not shared best-quality weights.')
    atomic(ROOT/'summary.json',result)
    lines=['# 四项缺口实现、训练及性能验收','','四项机制已实现，两个匹配训练组均完成2epochs×8样本、每样本四个两步监督事件。骨干冻结，全部有效稀疏参数训练。结果有质量均值收益，但严格质量线未过，理论速度未兑现，不能宣布最终方案成功。','','## 四项改动','',
'1. 独立teacher使用与学生相同338页硬分组保留算法和当条样本的参数；自行推进clean历史。学生监督完成后才提交teacher当前clean chunk，避免未来信息。另做共享学生历史的匹配控制。两组统一16probes、同初始化、同学习率/事件/更新次数。',
'2. 路由使用四个条件V编码。缓存rank8的页内子组KV交叉协方差对角项，估计query相关的精确减摘要残差；集合评分采用预计遗漏误差与候选贡献的对齐，并考虑选中/未选中页的不同增删项。这是低成本一阶对角近似，非精确oracle或任意query误差保证。',
'3. 四K代表重新中心化到真实页K均值；V的base保留真实页均值在rank8子空间的编码，学习零和Haar details。取消自由的head质量偏置，避免破坏基础质量；改为可学习残差评分门控。页局部、K均值、编码V均值、完整梯度及early/steady-state训练推理一致性均通过。',
'4. 历史保留完整K/V响应预处理；当前块去掉完整V响应，采用较廉价K特征聚合。增加不复制原始KV的current-summary图，直接读取已有当前cache；共享输出，释放未使用的旧参数，并计入预算。',
'', '原生版持久KV+aux 5,110,914,192B，图优化版5,112,819,024B，均低于允许上限5,112,856,576B（原上限+16MiB）。优化版仅剩37,552B预算余量。训练激活、teacher副本、路由中间矩阵/graph工作区另外计入峰值，不能混为历史缓存。',
'', '两个训练组各16次样本边界更新；全部有效参数均有有限非零梯度，骨干无梯度。每样本固定参数、下一样本重建缓存，避免缓存和权重版本混用。保留initial/latest/previous/epoch checkpoints，可按样本边界恢复；不保存QKV。',
'', '## 前段RGB质量','', '仅使用4个validation场景按预先规定的mean LPIPS+(1-SSIM)选checkpoint，初始化也可入选；两组均选epoch1。再使用6个test场景的新seed404/505进行确认。prompt此前出现过，不是全新prompt泛化证明。指标来自未压缩RGB[21,101)。', '', '| 方法 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---:|---:|---:|']
    for k,q in means.items():lines.append(f"| {k} | {q['lpips']:.6f} | {q['ssim']:.6f} | {q['psnr']:.3f} |")
    lines+=['', '共享历史训练平均优于独立历史训练，不能因为后者目标更完整就宣布更有效。四项组合没有做完整因子消融，不能将组合收益分别归因给每个模块。', '', '逐单元LPIPS不退化且SSIM不下降的既定保守门槛：']
    for a,bs in gates.items():
        for b,g in bs.items():lines.append(f"- {a} 对 {b}: {g['status']}；回退 {len(g['regressions'])}/12：{', '.join(g['regressions'])}。")
    lines+=['', '共享组对旧16probe仅飞鸟seed505失败，但并非纯舍入误差：LPIPS 0.285675→0.295828，SSIM 0.547339→0.520080。不能事后放宽门槛。', '', '## 完整30秒速度','', 'horse_field seed101，matched_e1，暖机6chunks及VAE，计入全部生成/缓存/路由/VAE时间。reference/graph/fast的两次完整latent逐元素一致且与前段验证一致。旧16probe和dense使用各自轨迹，不能声称它们与新模型视频相同。GPU为H200 NVL。', '', '| 实现 | 两次含VAE秒 | 平均秒 |','|---|---|---:|']
    for k,v in timings.items():lines.append(f"| {k} | {', '.join(f'{x:.3f}' for x in v['seconds'])} | {v['mean_seconds']:.3f} |" if v['mean_seconds'] else f'| {k} | pending | — |')
    lines+=['', 'fast相对本轮reference有所优化，但仍慢于dense和旧16probe；速度尚未兑现。主计时选择独立历史组，不把共享组的最好质量与另一模型的速度拼成联合通过。',
'', '激进torch.compile路由在初始化9chunk通过，但训练后完整视频前段检查失败，已拒绝晋级，不引用其耗时宣称成功。普通CUDA graph＋current-summary图已通过完整视频一致性。条件tail分块扫描无显著提升；进一步split-P在410页仅约1.03×合成收益且输出非逐元素一致，未部署。',
'', '## 理论成本与实现差距','', '稳态N=4608,H=12,P=410,page64,D128,keep41,R4,rank8,probe16。理想矩阵FLOPs：精确分支为dense的10%，条件tail约3.32%；16probe路由QK另约0.78%，不含特征、softmax、topk和聚合。90%精确页稀疏率不等于90%端到端计算减少。',
'', '实测设备校准：BF16矩阵约566.7TFLOPS，严格FP32约46.6TFLOPS，TF32约309.6TFLOPS；大buffer copy读写约4.02TB/s。这些是该microbenchmark的可达吞吐，不是所有形状都可达到的理论峰值。',
'', '| 模块 | 实测ms | 校准的乐观计算下界ms |','|---|---:|---:|','| dense attention | 2.793 | 1.311 |','| 精确稀疏attention | 0.272 | 0.131 |','| 条件tail | 0.681 | 0.169（近似计入TF32高低项与rank填充） |','| 路由QK | 0.181 | 0.125 |','| 完整路由，排除Python调度 | 1.908 | 无完整可比较下界 |',
'', '精确稀疏分支耗时约为dense的9.75%，与10%选页比例基本吻合；此处已经较好兑现。路由QK也相对接近该FP32校准下界，但完整路由约90.5%的耗时在QK之外，继续只减QK FLOPs不是主要解法。tail与乐观计算下界相差约4倍，含softmax/访存/小tile/依赖链，不能把全部差距归为可消除浪费，也没有硬件计数器证据证明接近极限。',
'', '当前pool cheap 0.464ms、full 0.612ms，cov另0.118ms；相同cov前提下current路径0.582对0.730ms，降低约20.3%。它仍然是相当大的在线开销。主要剩余工作是安全融合条件V/集合评分与当前聚合、限制中间张量流量，而不是宣称数学计算量本身已不可能或盲目提高稀疏率。',
'', '## 产物与边界','', '实现：gap_model.py、gap_route.py、gap_runtime.py、gap_train.py；精确优化：gap_graph_runtime.py、gap_fast_runtime.py、gap_cached_ops.py。独立训练对照、12个新seed单元质量、同模型完整性能及校准计算量已完成。新的机制并不意味着四种误差都被彻底消除：硬排序与STE近似、对角协方差残差、rank8子空间损失及剩余场景回退仍存在。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    atomic(ROOT/'status.json',dict(status='experiments_complete',quality_passed=False,theoretical_speed_realized=False,training_arms=2,updates_per_arm=16,test_units=48))
if __name__=='__main__':main()
