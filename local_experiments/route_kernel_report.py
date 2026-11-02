import json,statistics
from pathlib import Path
from .learned_cache.io import atomic
ROOT=Path('artifacts/route_kernel_20261031')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['exact','vector'],default='exact');variant=ap.parse_args().variant
    checks=read(ROOT/variant/'checks.json');roof=read(ROOT/'roofline.json');status=read(ROOT/variant/'acceptance/status.json')
    results=[json.loads(p.read_text()) for p in sorted((ROOT/variant/'acceptance').glob('*/*.json'))]
    lines=['# 路由实现优化：精度、速度与理论下限','','本轮保持训练后29层独立路由参数、原始RoPE、90%稀疏调用页预算和未选页补偿不变。比较对象是上一轮CUDA Graph路由；没有重新训练或调整页选择目标。','','## 最终实现','','最终入口：`local_experiments.route_exact_runtime.ExactRuntime`。','', '- 用共享CUDA Graph覆盖当前摘要、QK、K方差、归一化和路由后处理；按层读取指针别名，避免按层复制历史KV。','- 四代表logsumexp融合为一个kernel，匹配原生缩放、指数和相邻加法顺序。','- 将页特征构造融合，显式匹配原生相邻归约树和精确除法；标准差、最终打分GEMV保留原生实现，避免临界同分改页。','- 换页及LUT排序融合为一个kernel，保持own页保护与最小索引的同分处理。','- 当前V方差保留原BF16输入pool kernel；把输入先转FP32再用同一公式曾改变极少量BF16量化结果，未采用。','','这是推理路径；现有全层训练反向仍使用原训练实现，没有替换其STE或梯度路径。','','## 验证与边界案例','','较激进的全融合打分在前段3248次检查通过，但完整篮球视频在chunk23/layer16/head4/tile37遇到两候选原生同分：约一个FP32 ULP改变了删除页。该版本被拒绝，不作为交付速度。定位记录见debug/first_divergence.json。修正版保留原生打分并恢复特征计算顺序。']
    timing={}
    if checks:
        rows=checks['rows'];ts=checks['timings']
        timing={m:statistics.median(x['wall_ms'] for x in ts if x['method']==m) for m in ['graph','fused']}
        lines += ['',f'最终全长逐调用检查：{checks["events"]}次，覆盖训练前/训练后两套权重、篮球/蝴蝶两个场景，每段41chunks。选页及K代表/V code一致。最终打分不同元素总数：{sum(x.get("score_differences",0) for x in rows)}。',f'特征最大差异：{max(x.get("max_feature_error",0) for x in rows):.3g}；若仅有非规格化极小值差异，也要求最终打分一致，不仅检查最终页集合。',f'同次交替完整route微测中位数：{timing["graph"]:.4f} → {timing["fused"]:.4f} ms，{timing["graph"]/timing["fused"]:.2f}×。包含当前摘要、QK及全部后处理；不含图构建。']
    else:lines+=['','最终全长逐调用检查尚未完成，以exact/checks.json为准。']
    lines+=['','## 端到端验收','','训练后epoch1固定权重；seed1001，两个场景；每场景两轮，顺序graph→优化、优化→graph。41chunks/480frames，包括KV维护、路由、模型和VAE；排除加载、图捕获、预热、编码和指标。原生baseline只保存一份全长latent，优化结果逐元素比较；不重复保存QKV或编码视频。',f'状态：{status}','','| 场景 | 轮次 | 实现 | 生成+VAE秒 | 纯生成秒 | 峰值allocated GiB | 未观测到其他GPU进程 |','|---|---:|---|---:|---:|---:|---|']
    for r in results:
        p=r['performance'];lines.append(f'| {r["case"]} | {r["rep"]} | {r["method"]} | {p["total_generation_vae_seconds"]:.3f} | {p["generation_seconds"]:.3f} | {p["peak_allocated_bytes"]/2**30:.3f} | {r["telemetry"]["no_observed_foreign_processes"]} |')
    lines+=['','表中峰值allocated随运行顺序上升，不能把相邻行差值全部当成实现的净显存增量。历史KV预算按张量逐项核算；计算工作区的纯增量需独立进程测量。']
    comparisons=[]
    for case in ['basketball_court','butterfly_garden']:
        a=[r['performance']['total_generation_vae_seconds'] for r in results if r['case']==case and r['method']=='graph'];b=[r['performance']['total_generation_vae_seconds'] for r in results if r['case']==case and r['method']=='whole']
        if len(a)==len(b)==2:
            aa=statistics.mean(a);bb=statistics.mean(b);comparisons.append(dict(case=case,graph=aa,optimized=bb,speedup=aa/bb,saved=aa-bb));lines+=['',f'{case}两轮平均：{aa:.3f}→{bb:.3f}s，{aa/bb:.3f}×，节省{aa-bb:.3f}s。']
    lines+=['','全长latent一致才复用此前前段RGB质量结果；这次实现优化不声称解决90%稀疏质量仍未完全达标的问题。','','## 理论与可兑现收益','','稳态QK规模：12 heads × 288 probes × 1640代表 × D128，约1.451 GFLOP。H200 NVL按132 SM、每SM128 FP32单元和最大1785MHz计算，理想FP32峰值约60.32 TFLOP/s，仅QK的乐观计算下限约24.05μs。这个下限忽略归一化、选页、摘要、访存和发射，不是完整路由的可达时间。']
    if roof:
        s=roof['stages'];lines+=['',f'同机独立原语微测：QK {s["qk_bmm"]["median_ms"]*1000:.1f}μs，K方差 {s["k_variance"]["median_ms"]*1000:.1f}μs，原生logsumexp {s["native_logz"]["median_ms"]*1000:.1f}μs，融合logsumexp {s["fused_logz"]["median_ms"]*1000:.1f}μs，原生top-k {s["topk"]["median_ms"]*1000:.1f}μs。流式copy有效读写带宽约{roof["empirical_copy_bytes_per_second"]/1e12:.2f}TB/s，当前K/V至少读取28.31MB；实际还包括中间量与重复访问。流式copy与路由的缓存条件不同，不能直接视为路由可达带宽。','', '不能把24μs当成完整路由的理论下限后宣称已经接近它。当前仍有多个原生归约/选页/GEMV kernel，距离全融合的硬件极限有差距。精确参考算术也限制了重排、TF32和并行归约的使用。']
    if timing:
        saved=(timing['graph']-timing['fused'])*4524/1000;remaining=timing['fused']*4524/1000
        lines+=['',f'按每段4524次稀疏调用，单稳态微测推算本轮可节省约{saved:.2f}s，剩余路由总时间约{remaining:.2f}s。这是孤立微测外推的量级，不是严格端到端上限；前几chunk形状较小，CPU/GPU重叠也不同。不能把它当成已经测得的pipeline路由占比，最终看端到端表格。']
    lines+=['','额外探索了64位bitonic top-k：稳态约51→38μs（含不同边界的单独实验），但其同分规则不同于原生top-k；最大收益仅约0.06秒/视频，未在最终路径替换。下一步若继续kernel工作，应优先检查原生QK/小归约的融合收益，不能为极小收益再次引入未验证的同分变化。','','## 资源与恢复','','最终历史KV和全部新增输入辅助量5,096,066,228B，低于5,096,079,360B，余13,132B。包括图输入复制、各层指针表、复制的basis/center/weight及层索引。FP32转换结果、packed-K、logits、特征和图临时张量是计算工作区，物理显存额外占用，已在峰值allocated中报告；它们不增加可保留的历史。','','实验目录限1GiB、磁盘空闲至少50GiB；每个完整视频原子提交并可跳过重跑。恢复命令（先确认旧进程已停止）：','','```bash','CUDA_VISIBLE_DEVICES=4 bash local_setup/run.sh -m local_experiments.route_exact_accept > /tmp/route_exact_accept.log 2>&1','```','','代码身份与源码快照位于exact/acceptance/identity.json和exact/source_snapshot。失败版本及诊断保留作证据，不将其结果混入最终验收。']
    lines=[line.replace('exact/',variant+'/').replace('route_exact_accept','route_'+variant+'_accept').replace('route_exact_runtime.ExactRuntime','route_'+variant+'_runtime.'+('VectorRuntime' if variant=='vector' else 'ExactRuntime')) for line in lines]
    if variant=='vector':lines.insert(9,'- 特征写回改为合并访存：计算次序不变，单kernel实测约69→39μs。')
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n');atomic(ROOT/'summary.json',dict(variant=variant,status=status,check_passed=bool(checks and checks['passed']),timing=timing,comparisons=comparisons,units=len(results)))
if __name__=='__main__':main()
