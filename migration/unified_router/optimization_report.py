"""Merge controlled timings, matched Val quality and graph-aware kernel profiles."""
import argparse,csv,gzip,json,statistics,subprocess
from pathlib import Path
import numpy as np
from .common import config,atomic_json,sha


def csv_write(path,rows):
    with path.open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def attribution(folder):
    data=json.loads(gzip.decompress((folder/'trace.json.gz').read_bytes()))['traceEvents']
    scopes=[e for e in data if e.get('cat')=='user_annotation'and e.get('name','').startswith('OPT::')]
    launches={e['args']['correlation']:e for e in data if e.get('cat')in ['cuda_runtime','cuda_driver']and 'correlation'in e.get('args',{})}
    rows={};kernels=[]
    for e in data:
        if e.get('cat')!='kernel':continue
        api=launches.get(e.get('args',{}).get('correlation'));name='unattributed'
        if api:
            candidates=[s for s in scopes if s['tid']==api['tid']and s['ts']<=api['ts']<=s['ts']+s['dur']]
            if candidates:name=min(candidates,key=lambda s:s['dur'])['name'].replace('OPT::','')
        if name=='block_total':name='other_backbone_and_diagnostics'
        row=rows.setdefault(name,dict(module=name,kernel_ms=0.,kernels=0));row['kernel_ms']+=e['dur']/1000;row['kernels']+=1
        kernels.append(dict(module=name,kernel=e['name'],milliseconds=e['dur']/1000,
            registers=e['args'].get('registers per thread'),occupancy=e['args'].get('est. achieved occupancy %')))
    total=sum(r['kernel_ms']for r in rows.values())
    raw=json.loads((folder/'modules.json').read_text())
    assert abs(total-raw['actual_kernel_sum_ms'])<1e-6
    raw['cuda_scope_time_incomplete_for_graph_and_triton']=True
    raw['use_for_module_cuda_time']='kernel_attribution.json / modules.csv: API-correlation reconstruction'
    atomic_json(folder/'modules.json',raw)
    assert 'unattributed'not in rows,rows
    result=dict(total_kernel_ms=total,modules=sorted(rows.values(),key=lambda r:-r['kernel_ms']),
        rule='Each GPU kernel -> CUDA runtime/driver correlation -> innermost CPU OPT scope; CUDA graph nodes map to their graph launch. No GPU annotation spans or duplicated operator totals.',kernels=kernels)
    atomic_json(folder/'kernel_attribution.json',result);csv_write(folder/'modules.csv',result['modules']);csv_write(folder/'kernels.csv',kernels)
    return result


def report(root):
    out=root/'optimization';seal=json.loads((out/'final_manifest.json').read_text())
    for p,digest in {**seal['code'],**seal['unmodified_original']}.items():assert sha(p)==digest,p
    speed={};timings=[]
    for d in sorted((out/'final_benchmark').iterdir()):
        doc=json.loads((d/'timings.json').read_text());rows=doc['records'][1:]
        assert len(rows)==3 and len({r['latent_sha256']for r in doc['records']})==1
        name=d.name.replace('-replica','')
        for r in rows:
            timings.append(dict(variant=name,replica=d.name,**r))
        speed.setdefault(name,[]).extend(rows)
    table=[]
    for name,rows in speed.items():
        table.append(dict(variant=name,n=len(rows),generation_s=statistics.mean(r['generation_seconds']for r in rows),
            decode_s=statistics.mean(r['decode_seconds']for r in rows),total_s=statistics.mean(r['total_seconds']for r in rows),
            generation_std=statistics.stdev(r['generation_seconds']for r in rows),
            max_allocated_GiB=max(r['peak_allocated_GiB']for r in rows)))
    speed={r['variant']:r for r in table};csv_write(out/'speed_summary.csv',table);csv_write(out/'timing_samples.csv',timings)
    dense_hash=json.loads((out/'final_benchmark/Dense/timings.json').read_text())['records'][1]['first_chunk_sha256']
    assert all(r['first_chunk_sha256']==dense_hash for r in timings)
    assert len({r['latent_sha256']for r in timings if r['variant']=='Fast-K32'})==1
    cases=sorted((out/'quality_seed0').glob('*/result.json'));assert len(cases)==8
    rows=[];qualities={};rng=np.random.default_rng(20261024)
    for f in cases:
        doc=json.loads(f.read_text())
        for oldkind,name in [('K32-Set','Old-K32'),('V32-Set','Old-V32')]:
            d=json.loads((root/'videos'/f.parent.name/(oldkind+'_metrics.json')).read_text())
            q=d['quality']['sparse_frames_only'];rows.append(dict(case=f.parent.name,variant=name,lpips=q['lpips'],psnr=q['psnr_db'],keep=d['budget']['token_weighted_keep'],delta_lpips_vs_old=0.))
        for name,d in doc['variants'].items():
            oldkind='K32-Set'if 'K32'in name else'V32-Set'
            original=json.loads((root/'videos'/f.parent.name/(oldkind+'_metrics.json')).read_text())
            q=d['quality_vs_dense']['sparse_frames_only']
            rows.append(dict(case=f.parent.name,variant=name,lpips=q['lpips'],psnr=q['psnr_db'],keep=d['weighted_keep'],
                delta_lpips_vs_old=q['lpips']-original['quality']['sparse_frames_only']['lpips']))
    csv_write(out/'quality_cases.csv',rows)
    for name in sorted({r['variant']for r in rows}):
        rr=[r for r in rows if r['variant']==name];diff=np.array([r['delta_lpips_vs_old']for r in rr])
        bootstrap=diff[rng.integers(0,8,(10000,8))].mean(1)
        qualities[name]=dict(lpips=statistics.mean(r['lpips']for r in rr),psnr=statistics.mean(r['psnr']for r in rr),
            keep=statistics.mean(r['keep']for r in rr),delta_lpips=float(diff.mean()),
            ci95=[float(x)for x in np.quantile(bootstrap,[.025,.975])],better_cases=int((diff<0).sum()),cases=8)
    profiles={str(s):attribution(out/f'profile_q{s}')for s in [1,16]}
    correctness=json.loads((out/'correctness.json').read_text())
    original=json.loads((root/'speed_profile/summary.json').read_text())
    summary=dict(speed=speed,quality=qualities,profiles=profiles,checks=correctness,
        source_immutable=True,all_repeats_bit_identical=True,first_chunk_matches_dense=True,
        quality_scope='All8 Val prompts, seed0, sparse RGB frames9:45, same original latents and uncompressed VAE output. Not Test; does not prove perceptual equivalence.')
    atomic_json(out/'summary.json',summary)
    lines=['# SlimAV v10 执行优化与速度—质量验证','',
        '已完成独立优化适配器、24 次受控计时、8 个 Val prompt × 5 个新方案的视频回归，以及两组 CPU/CUDA profile。原训练权重、原网络代码、旧视频和旧指标均未修改。',
        '', '**结论：逐 query K32/V32 的生成时间从约 142/140 秒降到约 10.8/10.9 秒，约 13 倍加速；仍未超过同期 Dense 的约 4.52 秒。共享 query 的近似方案接近 Dense，但出现质量代价，也未取得可靠净加速。**',
        '', '## 1. 受控端到端速度','',
        '单张 A100-SXM4-40GB；相同首个 Val prompt、seed0、45 帧 480×832、4 chunks、4 去噪步、BF16 主干、FP32 路由、TF32 关闭。每个 GPU 一次完整预热后计时三次；K32 在两张 GPU 重复，共六次。编译、权重加载、T5 文本编码、LPIPS 和 MP4 编码不计入；总时长包含 VAE、RGB 回传及原流程的上下文重建。所有新计时结束后才开始质量回归和 profile。',
        '', '|方案|生成 / s|VAE及回传 / s|总计 / s|样本数|峰值 allocated / GiB|','|---|---:|---:|---:|---:|---:|']
    order=['Dense','U-Sol','Fast-K32','Fast-V32','Q16-K32','Q16-V32','Q64-K32']
    for name in order:
        r=speed[name];lines.append(f"|{name}|{r['generation_s']:.3f}|{r['decode_s']:.3f}|{r['total_s']:.3f}|{r['n']}|{r['max_allocated_GiB']:.2f}|")
    lines+=['','旧版生成 / 总计：U-Sol 40.617 / 42.534 s，K32 141.690 / 143.617 s，V32 140.240 / 142.157 s。同期 Dense 4.515 / 6.433 s，与本次新测 Dense 一致。详见 [旧速度报告](../speed_profile/REPORT.md)。',
        '', 'Fast 保留逐 query 决策规则；Q16/Q64 使用每组第一个 query 的路由结果覆盖该组，属于单独的算法近似。它们没有重新训练。所有方案首 chunk、layer0 和上下文重建仍为 Dense。',
        '', '## 2. 实施的优化','',
        '- 原型投影移出每个 128-query 循环，每个 block 只计算一次；同一历史缓存的原型跨去噪步复用。历史 P8/P4/P2 与当前 P16 保持原设计。',
        '- 路由执行批量扩大到 512，但候选探索哈希仍按原来的 128-query 规则计算，最后 72-query 尾块也保留。',
        '- 用稳定最小值选择的 Triton kernel 替代多轮完整排序及填洞操作，保留同分时页编号的顺序。',
        '- 编译并融合 score、threshold、原型构造；用独立 Triton kernel 融合 24 维特征、已训练 MLP、合法动作 argmax 和 mask 更新。没有改权重或开放 TF32。',
        '- CUDA Graph 按历史长度缓存并跨层复用；每次复制当前参数及输入，GPU 内部完成路由。取消逐 tile 的 CPU 标量读取，诊断结果最后统一回传。',
        '- 新 attention 按 query 组构建 KV 页并集，跳过全组不可见的页，使用 online softmax；组内仍逐 query 应用原 mask。4680-token chunk 的 8-token 尾页单独正确寻址。无需展开完整 token mask，也不搬运一份每-query KV 副本。',
        '', '复用 PyTorch/cuBLAS、Inductor、CUDA Graph 与 Triton。注意力算法参考 [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) 的 online-softmax 方法；本地 kernel 自行实现，未直接复制第三方源文件。',
        '', '也调研和实测了公开实现：[MIT Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) 的公开接口以 128-token block mask 为基础，不能直接表达原逐 query 的 64-token 页；[FlexAttention](https://pytorch.org/blog/flexattention/) 可用 mask callback 表达，但本机 PyTorch 2.6 的 32/64-query sparse-block 配置受到默认内核尺寸检查限制，成功运行的 128-query 配置在该真实 block 上约 12.52 ms，慢于自写内核。这个单配置结果不是对 FlexAttention 所有版本/调优组合的否定。详见 [组件实测](components.json)。',
        '', '## 3. 模块 profile 与剩余瓶颈','',
        '取真实第 4 chunk、step3、layer24，4680 个 query、18720 个 KV token。以下按 CUDA API correlation 把每个 GPU kernel 归到发起它的 CPU scope；CUDA Graph 内各节点归到 graph launch。每个 kernel 只计一次。直接使用 key_averages 的 CPU scope CUDA 时间会漏记 graph/Triton kernel，因此没有用漏记值作结论。',
        '', '|模块（GPU kernel 累计 ms）|Fast-K32|Q16-K32|','|---|---:|---:|']
    by={s:{r['module']:r['kernel_ms']for r in d['modules']}for s,d in profiles.items()}
    names=[('current_prototypes','当前原型'),('prototype_projection','原型投影'),('route_graph_copy_and_replay','路由（含输入复制 kernel）'),
        ('sparse_attention_including_union','稀疏注意力及页并集'),('ffn','FFN'),('cross_attention','交叉注意力')]
    for key,label in names:lines.append(f"|{label}|{by['1'].get(key,0):.3f}|{by['16'].get(key,0):.3f}|")
    lines.append(f"|整层 kernel 合计|{profiles['1']['total_kernel_ms']:.3f}|{profiles['16']['total_kernel_ms']:.3f}|")
    lines+=['', '原实现同一层的 masked SDPA 约 48.71 ms；新逐 query attention 约 3.4 ms，共享 Q16 后约 1.4 ms，Dense FlashAttention 约 3.0 ms。因此 attention 局部确实得到了计算收益，但新增路由仍抵消端到端收益。',
        '', '真实连续 query 的一个实测 block：逻辑保留率约 14%，逐 query 选择在 16-query kernel 上的页并集约 35%，32-query 并集约 44%。共享 Q16 后并集约 14%。这些是该 block 的实测值，不是全模型平均。',
        '', '更重要的是，路由仍要为所有页构造 32 维 content 和集合统计。第 4 chunk 有 296 页、2220 个原型；完整 content 张量约 12×4680×296×32×4 = 2.13 GB（实际分批产生），带来显著写入/读取开销。score 和 content 的主要乘法至少约 24 GFLOP，使用 FP32；Dense attention 使用 BF16 Tensor Core，不能仅按“保留 15%”推算总耗时。',
        '', '首 chunk、layer0、Dense context rebuild、FFN、QKV/输出投影、交叉注意力和 VAE 也不随稀疏率降低；VAE 本身约 1.9 秒。',
        '', '## 4. 相同 8 个 Val prompt 的视频质量','',
        '仅 seed0，与原 seed0 逐 prompt 配对；只统计首个 Dense chunk 之后的 36 帧。此处旧版均值不能与此前 16 cases 的均值直接混比。LPIPS 越低表示越接近 Dense，并非人类绝对画质评分。',
        '', '|方案|LPIPS↓|PSNR↑|KV保留率|LPIPS − 对应旧版（95% bootstrap CI）|','|---|---:|---:|---:|---|']
    for name in ['Old-K32','Fast-K32','Q16-K32','Q64-K32','Old-V32','Fast-V32','Q16-V32']:
        r=qualities[name];ci=r['ci95'];delta='—'if name.startswith('Old')else f"{r['delta_lpips']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]"
        lines.append(f"|{name}|{r['lpips']:.5f}|{r['psnr']:.3f}|{r['keep']:.2%}|{delta}|")
    lines+=['', '逐 query 优化版的均值接近旧版，但不能称为逐像素无损：浮点运算与 BF16 attention 归约顺序改变，经多层多步会积累。Q16/Q64 已明显改变逐 query 的选择，部分场景的结构和轨迹偏差更大，不建议直接替换原质量方案。bootstrap 以 prompt 为单位抽样 10000 次；8 个 prompt 的区间较宽，未访问 Test，也未据此重选 checkpoint。',
        '', f"正确性检查：9 个边界案例覆盖同分排序、全选、单页、混合页及 17/72/128 个 query；3 层×3 历史长度×2 head 组合，共 {correctness['total_real_queries']} 个真实 query 中，有 {correctness['total_changed_queries']} 个最终 mask 与参考不同。更长真实连续 query 的独立检查发现过 1/56160 个边界 query 差异，因此不宣称 mask 永远逐 bit 相同。所有新方案首 chunk latent/RGB 完全相同；计时重复及跨 GPU K32 replica 完全一致；两组 profiler 不改变生成 latent。", 
        '', '## 5. 代码、数据与复现','',
        '- [入口适配器](../../../../../migration/unified_router/fast_network.py)、[路由](../../../../../migration/unified_router/fast_route.py)、[候选/注意力 kernel](../../../../../migration/unified_router/fast_kernels.py)、[特征与 MLP kernel](../../../../../migration/unified_router/fast_features.py)、[CUDA Graph](../../../../../migration/unified_router/fast_graph.py)。',
        '- [源代码/实验冻结记录](final_manifest.json)、[原始计时 CSV](timing_samples.csv)、[逐 case 质量 CSV](quality_cases.csv)、[正确性检查](correctness.json)、[汇总 JSON](summary.json)。',
        '- [Fast 模块 CSV](profile_q1/modules.csv)、[Q16 模块 CSV](profile_q16/modules.csv)、[Fast kernel 明细](profile_q1/kernels.csv)、[Fast Perfetto trace](profile_q1/trace.json.gz)、[Q16 trace](profile_q16/trace.json.gz)。',
        '- [K32 全 8 场景视频](ALL8_Dense_OldK_FastK_Q16K_Q64K.mp4)：从左到右 Dense、原 K32、优化逐 query K32、Q16 K32、Q64 K32。',
        '- [V32 全 8 场景视频](ALL8_Dense_OldV_FastV_Q16V.mp4)：Dense、原 V32、优化逐 query V32、Q16 V32。',
        '- [第 40 帧全场景对比](frame40_comparison.png)、[速度—质量图](speed_quality.png)。',
        '', '使用优化版（默认不启用共享 query 近似）：','', '```python',
        'from migration.unified_router.fast_network import load_fast',
        'net, config = load_fast(experiment_directory, graph=True, fused=True,',
        '    history_cache=True, compiled_build=True, query_stride=1)',
        "net.set_mode('V32-Set')  # 或 K32-Set / U-Sol / Dense",
        '# 后续继续使用原 trajectory(...)，置于 torch.no_grad() 与原 sdpa_backend() 下。',
        '```','',
        '命令行复现见 `python -m migration.unified_router.fast_bench --help`，参数 `--graph --fused --history-cache --compiled-build` 对应本次 Fast；`--query-stride 16` 是独立近似。进度：`./v10_optimization_progress.sh`。适配器限定本实验的 batch1、12 heads、128 维、4680-token chunk、64-token 页；CUDA Graph 静态缓冲用于顺序推理，不提供并发调用保证。',
        '', '## 6. 下一步的优先顺序','',
        '1. 优先继续消除逐 query 路由的全页 content 中间张量：把原型加权、集合统计、候选页向量提取融合成流式归约，避免先写出全页 32 维向量再读回。需要重新验证数值和视频；当前 kernel 的寄存器压力仍较高。',
        '2. 若修改算法并重新训练，把训练目标对齐实际 GPU query block。用 8/16-query 共用页、实际原始 128 维 attention 误差训练，预算同时考察页并集和真实耗时；保留 first-chunk Dense、P16 current 与距离衰减历史。当前直接用单 query head 覆盖整组的质量代价说明不能只在推理时硬套。',
        '3. 精度优化（例如多项式高精度 Tensor Core 分解）应作为独立数值实验；本次未放开普通 TF32。更换公开稀疏库/升级 PyTorch 应在独立环境验证，不覆盖已有训练和评估环境。',
        '', '当前可采用逐 query Fast 版显著缩短研究评估时间；在获得新训练与更充分的质量证据前，Q16/Q64 保留为探索结果。没有将任何新方案自动改为原实验默认入口。','']
    # Use real absolute links for the source files instead of fragile relative depth.
    code_root=Path(__file__).resolve().parent
    text='\n'.join(lines)
    for f in ['fast_network.py','fast_route.py','fast_kernels.py','fast_features.py','fast_graph.py']:
        text=text.replace('../../../../../migration/unified_router/'+f,str(code_root/f))
    (out/'REPORT.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    labels=['Dense','Fast K32','Fast V32','Q16 K32','Q16 V32','Q64 K32']
    keys=['Dense','Fast-K32','Fast-V32','Q16-K32','Q16-V32','Q64-K32']
    gen=[speed[k]['generation_s']for k in keys];dec=[speed[k]['decode_s']for k in keys]
    axes[0].barh(labels,gen,label='Generation');axes[0].barh(labels,dec,left=gen,label='VAE + transfer',color='#95b7c7')
    axes[0].invert_yaxis();axes[0].set_xlabel('Seconds / 45-frame video (single A100)');axes[0].legend()
    for i,k in enumerate(keys):axes[0].text(gen[i]+dec[i]+.1,i,f"{gen[i]+dec[i]:.2f}",va='center',fontsize=9)
    axes[0].set_xlim(0,15);axes[0].set_title('Warm end-to-end latency')
    for k in ['Fast-K32','Fast-V32','Q16-K32','Q16-V32','Q64-K32']:
        axes[1].scatter(speed[k]['total_s'],qualities[k]['lpips'],s=60)
        axes[1].annotate(k,(speed[k]['total_s'],qualities[k]['lpips']),xytext=(4,5),textcoords='offset points',fontsize=8)
    axes[1].axhline(qualities['Old-K32']['lpips'],linestyle='--',color='gray',label='Original K32 quality')
    axes[1].set_xlabel('Total seconds (lower is faster)');axes[1].set_ylabel('LPIPS vs Dense (lower is closer)')
    axes[1].set_xlim(6.2,14.5)
    axes[1].set_title('Matched 8 Val prompts, seed0');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'speed_quality.png',dpi=160);fig.savefig(out/'speed_quality.pdf');plt.close(fig)
    atomic_json(out/'verification.json',dict(passed=True,original_and_new_sources_unchanged=True,
        benchmark_repeats=24,quality_cases=8,quality_variants=5,profile_kernel_attribution_complete=True,
        masks_bit_identical_claim=False,decoded_video_bit_identical_claim=False))
    print(json.dumps(dict(speed=speed,quality=qualities),indent=2))


def videos(root):
    out=root/'optimization';cases=sorted((out/'quality_seed0').glob('*/result.json'))
    for name in ['Dense_OldK_FastK_Q16K_Q64K','Dense_OldV_FastV_Q16V']:
        paths=[f.parent/(name+'.mp4')for f in cases];listing=out/(name+'.concat.txt')
        # MP4 container duration rounds 45/16 to milliseconds. Supply the exact
        # frame duration to avoid tiny inter-clip timestamp gaps during concat.
        listing.write_text(''.join("file '"+str(p.resolve())+"'\nduration 2.8125\n"for p in paths))
        subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(out/f'ALL8_{name}.mp4')],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root,_=config(a.directory)
    report(root);videos(root)

if __name__=='__main__':main()
