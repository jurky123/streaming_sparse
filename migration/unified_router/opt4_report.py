"""Final IO-aware route report: controlled speed, quality, profiles and limits."""
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
from .optimization_report import attribution


def read(p):return json.loads(p.read_text())


def csv_write(path,rows):
    with path.open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    base=Path(__file__).resolve().parents[2]
    root=base/'streaming_sparse/v10/experiments/full29_keep15_e2e_20261024';out=root/'optimization4'
    seal=read(out/'fused_manifest.json')
    digests={**seal['code'],**seal['prior_unchanged']}
    for file,digest in digests.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    validations=[]
    for rank in range(8):
        assert read(out/f'fused_validation/rank{rank}.json')['passed']
        validations+=read(out/f'fused_validation/results_rank{rank}.json')
    assert len(validations)==32 and all(r['latent_bit_identical']and r['rgb_bit_identical']for r in validations)
    samples=[];speed={};hashes={}
    for folder in sorted((out/'fused_benchmark').iterdir()):
        records=read(folder/'timings.json')['records'];rows=records[1:]
        assert len(rows)==3 and len({r['latent_sha256']for r in records})==1
        hashes[folder.name]=rows[0]['latent_sha256']
        speed[folder.name]=dict(variant=folder.name,n=3,
            generation_s=statistics.mean(r['generation_seconds']for r in rows),
            generation_std_s=statistics.stdev(r['generation_seconds']for r in rows),
            decode_s=statistics.mean(r['decode_seconds']for r in rows),
            total_s=statistics.mean(r['total_seconds']for r in rows),
            peak_allocated_GiB=max(r['peak_allocated_GiB']for r in rows))
        samples.extend(dict(variant=folder.name,**r)for r in rows)
    assert len(samples)==24 and len({r['first_chunk_sha256']for r in samples})==1
    assert hashes['Hybrid-K32']==hashes['Hybrid-K32-replica']
    quality=[];audit=[];cases=sorted((out/'quality').glob('*/result.json'));assert len(cases)==8
    for file in cases:
        doc=read(file);old=read(root/'optimization/quality_seed0'/file.parent.name/'result.json')['variants']
        for kind in ['K32','V32']:
            prior=old['Fast-'+kind]
            metrics=prior['quality_vs_dense']['sparse_frames_only']
            quality.append(dict(prompt=file.parent.name,variant='Opt2',mode=kind,lpips=metrics['lpips'],psnr_db=metrics['psnr_db'],
                weighted_keep=prior['weighted_keep'],delta_lpips_vs_opt2=0.,lpips_to_opt2=0.))
        for key,row in doc['variants'].items():
            metrics=row['quality_vs_dense']['sparse_frames_only'];prior=old['Fast-'+row['mode']]['quality_vs_dense']['sparse_frames_only']
            assert row['first_chunk_identical']
            quality.append(dict(prompt=file.parent.name,variant=row['variant'],mode=row['mode'],lpips=metrics['lpips'],psnr_db=metrics['psnr_db'],
                weighted_keep=row['weighted_keep'],delta_lpips_vs_opt2=metrics['lpips']-prior['lpips'],
                lpips_to_opt2=row['quality_vs_opt2']['sparse_frames_only']['lpips']))
            audit.extend(dict(prompt=file.parent.name,variant=row['variant'],mode=row['mode'],**r)for r in row['attention_audit'])
    qsummary={};rng=np.random.default_rng(20261024)
    for variant in ['Opt2','Hybrid','FP16']:
        for kind in ['K32','V32']:
            rr=[r for r in quality if (r['variant'],r['mode'])==(variant,kind)];assert len(rr)==8
            diff=np.array([r['delta_lpips_vs_opt2']for r in rr]);ci=np.quantile(diff[rng.integers(0,8,(10000,8))].mean(1),[.025,.975])
            worst=max(rr,key=lambda r:r['delta_lpips_vs_opt2'])
            qsummary[variant+'-'+kind]=dict(lpips=statistics.mean(r['lpips']for r in rr),psnr_db=statistics.mean(r['psnr_db']for r in rr),
                keep=statistics.mean(r['weighted_keep']for r in rr),delta_lpips=float(diff.mean()),delta_ci95=list(ci),
                lpips_to_opt2=statistics.mean(r['lpips_to_opt2']for r in rr),worst_prompt=worst['prompt'],worst_delta=worst['delta_lpips_vs_opt2'])
    asummary={}
    for variant in ['Hybrid','FP16']:
        rr=[r for r in audit if r['variant']==variant];assert len(rr)==144
        asummary[variant]=dict(blocks=len(rr),queries=sum(r['queries']for r in rr),changed_queries=sum(r['changed_queries']for r in rr),
            delta_nmse=sum(r['delta_sse']for r in rr)/sum(r['reference_energy']for r in rr),
            nmse_to_dense=sum(r['sse_to_dense']for r in rr)/sum(r['dense_energy']for r in rr),
            reference_nmse_to_dense=sum(r['reference_sse_to_dense']for r in rr)/sum(r['dense_energy']for r in rr))
    profiles={'Opt2':read(root/'optimization2/profile_guarded/kernel_attribution.json')}
    for variant in ['Hybrid','FP16']:
        folder=out/('profile_fused_'+variant.lower())
        assert read(folder/'modules.json')['observer_latents_bit_identical']
        profiles[variant]=attribution(folder)
    modules={k:{r['module']:r['kernel_ms']for r in p['modules']}for k,p in profiles.items()}
    kernels={}
    for name,p in profiles.items():
        table=collections.defaultdict(float)
        for row in p['kernels']:table[row['kernel']]+=row['milliseconds']
        kernels[name]=dict(table)
    summary=dict(selected_entry='migration.unified_router.opt4_fused.load_fused',speed=speed,quality=qsummary,
        actual_attention_audit=asummary,actual_attention_audit_source='optimization4/quality, before the final mask-read fusion',
        profiles={k:dict(total_kernel_ms=v['total_kernel_ms'],modules=v['modules'])for k,v in profiles.items()},
        final_fusion_bitwise_cases=32,quality_reference='Opt2 Guarded = saved Fast1 on these 16 prompt/head cases',
        precision_scope='Only routing dots; backbone, global PyTorch TF32 flag, budget, weights and prototypes unchanged',
        retrained=False,test_used=False,source_files_verified=len(digests))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    csv_write(out/'speed_summary.csv',list(speed.values()));csv_write(out/'timing_samples.csv',samples)
    csv_write(out/'quality_cases.csv',quality);csv_write(out/'attention_audit.csv',audit);csv_write(out/'fused_bitwise_checks.csv',validations)

    lines=['# SlimAV：IO-aware 路由、Gram 聚合与局部混合精度', '',
        '**最终实现不输出全页 32 维 content，也不输出已选集合的权重副本。FP16 版 K32/V32 生成约 7.05/7.15 秒，相比同期 Opt2 再减少约 11%；8 个 Val 场景的平均 LPIPS 变化小，但个别场景仍有可见轨迹差异。未重训。**', '',
        '最终入口是 `migration.unified_router.opt4_fused.load_fused`。`opt4.network.load_compact` 是融合前版本，仍保留作消融。原实验默认入口没有被覆盖。', '',
        '## 1. 受控端到端速度', '',
        '单张 A100-SXM4-40GB，45 帧 480×832，4 chunks×4 去噪步，相同首个 Val prompt 与 seed0。每个配置完整预热一次后测量三次；Hybrid-K32 另在第二张 GPU 重复三次。以下主表每项使用其主 GPU 的三次测量，replica 单独存于 CSV。所有计时完成后才运行质量验证及 profile。', '',
        '生成包括原上下文重建；总计包含 VAE 与 RGB 回传。加载、编译、文本编码、LPIPS 和 MP4 编码不计时。所有重复与 Hybrid 跨 GPU replica latent 一致，所有方案首 chunk 与 Dense 一致。', '',
        '|方案|生成 / s|VAE及回传 / s|总计 / s|峰值 allocated / GiB|', '|---|---:|---:|---:|---:|']
    order=['Dense','Opt2-K32','Hybrid-K32','FP16-K32','Opt2-V32','Hybrid-V32','FP16-V32']
    for key in order:
        r=speed[key];lines.append(f"|{key}|{r['generation_s']:.3f}|{r['decode_s']:.3f}|{r['total_s']:.3f}|{r['peak_allocated_GiB']:.2f}|")
    for kind in ['K32','V32']:
        a,b=speed['Opt2-'+kind],speed['FP16-'+kind]
        lines.append(f"\nFP16-{kind}：生成减少 {1-b['generation_s']/a['generation_s']:.2%}，总时长减少 {1-b['total_s']/a['total_s']:.2%}。")
    lines+=['', '峰值显存没有按中间张量缩减比例下降，因为整模型、KV、CUDA Graph 缓冲与 VAE 共同决定峰值。本轮主要减少执行期间的张量读写，不能把它宣传为整模型显存减半。仍慢于 Dense。', '',
        '## 2. 最终数据流：保留必要统计，消除全页向量', '',
        '设第 b 页的 P×32 投影原型为 V_b，页内 softmax 权重为 w_b，页成本为 t_b。原 content 为 c_b=w_bV_b。全体与已选集合需要的统计可以改写为：', '',
        '`G_b = V_b V_bᵀ`；`‖c_b‖² = w_b G_b w_bᵀ`',
        '`μ_S = (Σ_b∈S t_b w_b V_b) / (Σ_b∈S t_b)`',
        '`variance_S = (Σ_b∈S t_b ‖c_b‖²) / (Σ_b∈S t_b) − ‖μ_S‖²`', '',
        '1. 将四组历史/当前原型整理成连续布局，QK 打分合并为一次 GEMM。Gram 矩阵每个 block 构造一次，供该 block 的全部 query tiles 复用；当前实现没有跨去噪步缓存它。构造和复制均计入端到端及 route profile。',
        '2. 第一次页内扫描同时计算 score、归一化权重、Gram 二次型。避免先打分、后处理时重复 softmax。只保留 scalar score/energy 和一份原型权重。',
        '3. 由全页 score 计算阈值与候选。由于阈值需要全页均值/标准差，这个阶段屏障仍然存在。',
        '4. 将权重按 `[head, prototype, query]` 存储，使页内连续 query 的写入更连续。均值 GEMM 在读取时应用 mask，直接输出全体和已选两组均值；省去旧 `_select` kernel 和第二份权重缓冲。',
        '5. 只为最多 8 个删除候选＋8 个添加候选计算精确公式的 32 维向量，再运行原来的 81 动作 head。没有共用或广播 query 的 mask。', '',
        '这里的 query tile=64 只是 CUDA 内核并行粒度，和被否定的 Q64 共享路由不同：64 个 query 的 score、mask、统计与动作始终独立。', '',
        '## 3. 内存、精度与计算量', '',
        '代表层为 layer24、第4 chunk、step3：12 heads、4680 queries、296 页、2220 个原型。下表是整层累计产生的逻辑张量，十进制 MB；实际按512 queries分批，不是同时驻留的峰值或实测DRAM流量。', '',
        '|中间量|Opt2|最终版|', '|---|---:|---:|',
        '|全页32维content|2127.8 MB|0|',
        '|原型logits|498.7 MB|498.7 MB|',
        '|显式原型权重|不单独保留|249.4 MB，FP16一份|',
        '|已选集合权重副本|无此布局|0，由均值GEMM读取时应用mask|',
        '|页级energy|不单独保留|66.5 MB|',
        '|候选向量＋统计输出|候选从全页content读取|约133 MB|', '',
        '**尚未完全消除物化：仍保留 logits 和一份原型权重。** 原 content 的全量写出与统计读取约4.26 GB逻辑访问被移除，但新增Gram/权重/候选访问也必须计入。因此不能把4.26 GB直接当作净DRAM节省。', '',
        '|运算|Hybrid|FP16版|', '|---|---|---|',
        '|Q投影及原型投影|原FP32|原FP32|',
        '|QK打分乘法|局部TF32x3，FP32累加|FP16输入，FP32累加|',
        '|softmax、logsumexp、阈值|FP32|FP32|',
        '|Gram矩阵构造|FP32|FP32|',
        '|Gram二次型、均值GEMM、动作head首层乘法|FP16输入，FP32累加|FP16输入，FP32累加|',
        '|方差、归一化、GELU、head末层归约|FP32|FP32|', '',
        '全局 PyTorch TF32 仍关闭，主干执行不变。精度降低只在指定 Triton dot 中发生；TF32x3 是多次Tensor Core乘法组合的高精度近似，不应称为IEEE逐bit等价。参考：[Triton dot文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)。', '',
        '这个改写甚至可能增加乘法总量：原全页content主乘法约8 GFLOP，改成两套集合均值约16 GFLOP，另有Gram二次型。收益来自更合适的矩阵形状、局部Tensor Core计算和更少中间读写，不是简单减少FLOPs。A100 FP32与FP16/BF16 Tensor Core吞吐量差异也很大，不能用相同FLOP数直接估计耗时：[NVIDIA数据表](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf)。', '',
        '采用二阶矩方差时做非负截断，单页已选集合的方差显式设为0，避免消减误差留下伪方差。它与原中心化求和并非浮点逐bit等价，因此实际注意力误差和视频回归是采用依据。', '',
        '## 4. 最终 profile 与剩余热点', '',
        '以下把每个CUDA kernel经runtime/driver correlation唯一归属到CPU scope，CUDA Graph节点归到graph launch。统计的是kernel累计时间，不是CPU scope漏计值或墙钟时间。两个新profile观察器均不改变生成latent。', '',
        '|模块 / ms|Opt2|Hybrid最终|FP16最终|', '|---|---:|---:|---:|']
    for key,label in [('route_graph_copy_and_replay','路由及辅助数据构造'),('sparse_attention_including_union','稀疏attention与页并集'),
        ('ffn','FFN'),('cross_attention','交叉attention'),('current_prototypes','当前原型'),('prototype_projection','原型投影')]:
        lines.append('|'+label+'|'+ '|'.join(f'{modules[name][key]:.3f}'for name in ['Opt2','Hybrid','FP16'])+'|')
    lines.append('|整层kernel合计|'+'|'.join(f"{profiles[name]['total_kernel_ms']:.3f}"for name in ['Opt2','Hybrid','FP16'])+'|')
    lines+=['', 'FP16最终路由主要kernel：', '', '|kernel|累计 ms|', '|---|---:|']
    for k,v in sorted(kernels['FP16'].items(),key=lambda r:-r[1]):
        if k in ['_fused_first','_masked_mean','_mm','_candidate_actions','_short_candidates','_packed_candidates','_gram_stats']:
            lines.append(f'|{k}|{v:.3f}|')
    lines+=['', '15%逻辑保留率不代表路由只扫描15%页；页打分与全体集合统计依然覆盖全部页。attention还受到同一GPU query block页并集的影响。FFN、投影、上下文重建、首chunk Dense、layer0 Dense与约1.9秒VAE，也不随保留率下降。', '',
        '## 5. 真实注意力误差与32组视频', '',
        '检查8个注册Val prompt、seed0、K32/V32与两个新版本，共32组。Dense与Opt2使用原已冻结视频latent，Opt2在这些样本上已验证等于保存的Fast1。LPIPS/PSNR在未压缩VAE输出的后36帧计算；首9帧完全一致。未访问Test、未重选checkpoint。', '',
        '|方案|LPIPS对Dense↓|相对Opt2变化|95% prompt bootstrap区间|KV保留率|', '|---|---:|---:|---|---:|']
    for key in ['Opt2-K32','Hybrid-K32','FP16-K32','Opt2-V32','Hybrid-V32','FP16-V32']:
        r=qsummary[key];ci=r['delta_ci95'];lines.append(f"|{key}|{r['lpips']:.6f}|{r['delta_lpips']:+.6f}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|{r['keep']:.3%}|")
    lines+=['', '置信区间以8个prompt为抽样单位、bootstrap10000次；不是把帧或query当作独立质量样本。FP16-V32均值略好不能解释成低精度更准确，只能说明这些生成轨迹在该指标下更接近Dense。', '',
        '**局部误差很小，不代表视频逐像素一致。** FP16-K32的海豹场景LPIPS对Dense增加约0.0163；FP16-V32的Mumbai robot场景增加约0.0101。抽帧观察主要呈现姿态、位置及细节差异，不能据平均分保证所有场景等价。对比视频保留全部场景而非只挑较好例子。', '',
        '|同一Q/K/V上的局部审计|Hybrid|FP16|', '|---|---:|---:|']
    for label,field in [('抽查block数','blocks'),('head-query数','queries'),('最终mask变化query数','changed_queries'),
                         ('相对Opt2注意力输出NMSE','delta_nmse'),('实际注意力NMSE对Dense','nmse_to_dense'),('相同输入Opt2的NMSE对Dense','reference_nmse_to_dense')]:
        lines.append(f"|{label}|{asummary['Hybrid'][field]:.8g}|{asummary['FP16'][field]:.8g}|")
    lines+=['', '局部审计采自最后一次mask读取融合之前：每个版本在生成时抽查layer1/24/29、三个稀疏chunk、step3，共144个block。比较发生在新轨迹的同一份Q/K/V上，误差按输出能量汇总；两版本输入轨迹不同，不能把其对Dense小数差直接当成严格的版本优劣。最终融合另做完整视频一致性验证，结果见下段。', '',
        '**最后的mask读取融合额外验证了32组完整latent与解码RGB，全部逐bit等于前一轮已经评估的视频。** 因此本报告的视频和质量数据适用于最终`load_fused`。这条逐bit结论只针对最后一次融合，不是声称整个新方案与Opt2无损。', '',
        '## 6. 哪些尝试有用，哪些没有', '',
        '- 前轮纯FP32并行流式归约把中间缓冲从233 MB缩至46 MB，却因执行效率更差而变慢。省内存不能代替测速。',
        '- 本轮直接Tensor Core页块归约仍慢于Gram聚合，未采用；Gram将大量小的页内向量工作改成集合GEMM。',
        '- 初版Gram分别处理四组原型；连续布局合并后避免四次候选向量扫描，但仍重复softmax。',
        '- 最终将score/权重/energy合并，调整权重布局，再把选择mask融入均值GEMM。融合前FP16 K32/V32约7.13/7.23秒，最终约7.05/7.15秒，输出保持不变。',
        '- 普通TF32的单block误差大于FP16，速度也没有明显优势，未推进全视频；选择Hybrid与FP16做完整评估。初期使用旧候选kernel的测速已移到preliminary_probes，不能当成最终公平基线。', '',
        '## 7. 下一步如何继续优化，以及何时值得重训', '',
        '**最高优先级是进一步消掉logits和剩余原型权重，做两遍片上计算。** 第一遍分块QK得到每页score/energy；阈值和候选确定后，第二遍在均值kernel内重算QK，用已知页logsumexp恢复w，同时累加全体/已选两套均值。只让scalar score/energy、最终统计和16个候选向量落显存。', '',
        '这相当于用额外QK乘法与exp换掉logits/权重的写出和读回，更接近FlashAttention的IO思路。必须采用大query/prototype tiles、Tensor Core与合适的流水，不应回到之前低效的逐页FP32循环。每页score已知后，第二遍不需要重新做全局阈值。参考：[FlashAttention原论文](https://arxiv.org/abs/2205.14135)。这一步尚未实现，不能提前计入加速。', '',
        '其次是动作head：当前仍为每query装配81×24特征，约束在寄存器和SIMT操作上。可把11个左右的公共特征贡献提前计算，将归一化折入第一层参数，减少重复装配；若进一步减少候选数量或改成更便宜的集合特征，才可能产生更明显的额外收益。', '',
        '**重训应围绕最终便宜的特征与动作空间进行。** 本轮同输入实际AV误差几乎不变，没有证据必须仅为当前混合精度立即重训。但如果下一步用静态页中心近似全体统计、减少候选或改变特征，就应按以下方式适配：', '',
        '1. 复用现有Train/Val原始Q/K/V缓存；用新kernel重算特征和normalization，保持Test封闭。',
        '2. 若初始mask或候选改变，对新的81动作或缩减动作集重算原始128维attention的实际误差标签；不能无条件沿用旧动作标签。',
        '3. 只训练小router head，主干/LoRA不动。沿用当前`actual_av_nmse`目标：最小化硬动作实际AV NMSE的概率加权期望；best按Val硬选动作后的实际AV NMSE挑选，而不是特征拟合误差或浮点一致率。',
        '4. 训练时显式模拟部署精度与统计公式，并约束相同约15%预算；最后再做多seed端到端视频验证。局部AV变好也不保证固定seed视频更接近Dense。', '',
        '## 8. 结果与复现', '',
        '- [K32全部8场景视频](ALL8_Dense_Opt2_Hybrid_FP16_K32.mp4)、[V32全部8场景视频](ALL8_Dense_Opt2_Hybrid_FP16_V32.mp4)：从左到右Dense、Opt2、Hybrid、FP16，每个文件360帧、约22.5秒。',
        '- [差异较大场景抽帧](worst_case_frames.jpg)、[速度/质量/profile图](speed_quality_profile.png)。',
        '- [逐次计时](timing_samples.csv)、[逐场景质量](quality_cases.csv)、[同输入注意力审计](attention_audit.csv)、[最终融合32组逐bit检查](fused_bitwise_checks.csv)、[汇总JSON](summary.json)。',
        '- [最终FP16模块profile](profile_fused_fp16/modules.csv)、[kernel明细](profile_fused_fp16/kernels.csv)、[trace](profile_fused_fp16/trace.json.gz)。',
        '- [核心冻结记录](manifest.json)、[最终融合冻结记录](fused_manifest.json)。', '']
    for f in ['opt4_fused.py','opt4/streamgram.py','opt4/packed.py','opt4/route.py','opt4/actions.py','opt4/network.py','opt4/quality.py']:
        lines.append(f'- [{f}]({base}/migration/unified_router/{f})')
    lines+=['', '```python','from migration.unified_router.opt4_fused import load_fused',
        'net, config = load_fused(experiment_directory, method="streamgram",',
        '    precision="fp16", score_precision="fp16", tile=64)',
        '# Hybrid: score_precision="tf32x3"；其他参数相同。',
        'net.set_mode("V32-Set")  # 或 K32-Set',
        '# 使用原 trajectory，在 torch.no_grad() 与原 sdpa_backend() 下执行。','```','',
        '适配器默认Hybrid；原模型入口未改。限定当前batch1、12 heads、128维、4680-token chunk、64-token页、顺序CUDA Graph推理。首chunk与layer0 Dense；后续首chunk KV按普通历史处理；当前P16、历史P8/P4/P2、β和冻结权重均保持原实验。状态：`./v10_optimization4_progress.sh --once`。','']
    (out/'REPORT.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    ax=axes[0,0];gen=[speed[k]['generation_s']for k in order];dec=[speed[k]['decode_s']for k in order]
    ax.barh(order,gen,label='Generation');ax.barh(order,dec,left=gen,label='VAE + transfer',color='#a4c8ce')
    for i,k in enumerate(order):ax.text(gen[i]+dec[i]+.05,i,f'{gen[i]+dec[i]:.2f}s',va='center',fontsize=9)
    ax.invert_yaxis();ax.set_xlim(0,11.2);ax.set_xlabel('Seconds / 45-frame video');ax.legend(fontsize=8)
    ax.set_title('Final warm E2E latency, single A100')
    ax=axes[0,1];names=['Opt2','Hybrid','FP16'];left=[0.,0.,0.]
    for label,key in [('Routing','route_graph_copy_and_replay'),('Attention','sparse_attention_including_union')]:
        values=[modules[k][key]for k in names];ax.barh(names,values,left=left,label=label);left=[a+b for a,b in zip(left,values)]
    ax.barh(names,[profiles[k]['total_kernel_ms']-left[i]for i,k in enumerate(names)],left=left,label='Other')
    ax.invert_yaxis();ax.set_xlabel('Cumulative GPU kernel ms');ax.legend(fontsize=8);ax.set_title('Layer24, chunk4, step3')
    ids=sorted({r['prompt']for r in quality})
    for ax,kind in zip(axes[1],['K32','V32']):
        for variant in ['Hybrid','FP16']:
            rows={r['prompt']:r for r in quality if (r['variant'],r['mode'])==(variant,kind)}
            ax.plot(range(8),[rows[pid]['delta_lpips_vs_opt2']for pid in ids],marker='o',label=variant)
        ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(range(8),[pid[:4]for pid in ids]);ax.set_xlabel('Val prompt ID prefix')
        ax.set_ylabel('LPIPS vs Dense minus Opt2');ax.set_title(f'{kind}: per-prompt quality drift (seed0)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'speed_quality_profile.png',dpi=160);fig.savefig(out/'speed_quality_profile.pdf');plt.close(fig)
    verification=dict(passed=True,source_files_verified=len(digests),final_timing_samples=24,quality_cases=32,
        fusion_preserves_complete_latent_and_rgb_cases=32,same_qkv_audit_blocks=288,first_chunk_matches_dense=True,
        kernel_attribution_complete=True,weights_retrained=False,test_used=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(dict(verification=verification,quality=qsummary,speed=speed),indent=2))


if __name__=='__main__':main()
