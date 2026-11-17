"""Short-video speed and implementation-error acceptance for reused kernels."""
import json,os,math
from pathlib import Path
import numpy as np
import torch
from .weights import ROOT
from ..learned_cache.io import atomic,digest
from ..high_sparse_experiment import prefix_quality

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1/legacy_reuse';names=['baseline','rope_only','all','no_summary'];rows={};errors={}
    ref=torch.load(root/'baseline_samegpu/latents.pt',weights_only=True).float()
    assert torch.equal(ref,torch.load(root/'baseline/latents.pt',weights_only=True).float())
    rgb=np.load(root/'baseline_samegpu/pixels.npy',mmap_mode='r')
    import lpips
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache');metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    for name in [*names,'dense']:
        folder={'baseline':'baseline_samegpu','dense':'dense_samegpu'}.get(name,name)
        r=json.loads((root/folder/'result.json').read_text());d=json.loads((root/folder/'decode.json').read_text())
        if name=='baseline':r['stages']=json.loads((root/'baseline/result.json').read_text())['stages']
        rows[name]=dict(generation_seconds=r['median_seconds'],generation_trials=r['seconds'],vae_seconds=d['median_seconds'],total_seconds=d['generation_plus_decode_seconds'],stages=r['stages'])
        if name in ('baseline','dense'):continue
        z=torch.load(root/name/'latents.pt',weights_only=True).float();delta=z-ref
        pixels=np.load(root/name/'pixels.npy',mmap_mode='r');q=prefix_quality([pixels],[rgb],metric)
        sums=0.;square=0.;changed=0;maxabs=0
        for i in range(len(pixels)):
            x=pixels[i].astype(np.float64)-rgb[i].astype(np.float64);sums+=np.abs(x).sum();square+=(x*x).sum();changed+=np.count_nonzero(x);maxabs=max(maxabs,float(np.abs(x).max()))
        total=pixels.size
        errors[name]=dict(latent_bitwise_equal=torch.equal(z,ref),latent_nmse=float(delta.square().sum()/ref.square().sum()),latent_max_abs=float(delta.abs().max()),rgb_bitwise_equal=changed==0,rgb_mae_255=sums/total,rgb_rmse_255=(square/total)**.5,rgb_max_abs_255=maxabs,rgb_changed_fraction=changed/total,psnr=q['psnr'],ssim=q['ssim'],lpips=q['lpips'],metric_window=[21,101],rgb_scope='all 105 frames')
    for r in rows.values():
        r['generation_speedup_vs_baseline']=rows['baseline']['generation_seconds']/r['generation_seconds'];r['total_speedup_vs_baseline']=rows['baseline']['total_seconds']/r['total_seconds'];r['total_speedup_vs_dense']=rows['dense']['total_seconds']/r['total_seconds']
    micro={}
    for name,path in [('rope','rope'),('risk','risk'),('attention','attention432'),('summary','summary')]:
        r=json.loads((root/path/'result.json').read_text());which='legacy_compiled' if name=='risk' else 'legacy'
        before=r['rows']['current'];after=r['rows'][which]
        bt=before.get('steady_timing',before['timing'])['gpu_ms_median'];at=after.get('steady_timing',after['timing'])['gpu_ms_median']
        micro[name]=dict(before_ms=bt,after_ms=at,speedup=bt/at,nmse=after['nmse'],max_abs=after['max_abs'],theory=r['extra'])
    atomic(root/'acceptance.json',dict(performance=rows,implementation_errors=errors,micro=micro,configuration=dict(video='city_walk_s701',chunks=9,history_chunks=5,history_pages=360,frames=105,fps=16,resolution=[512,768]),scope='Weights and Full_Pset unchanged; only prior optimized non-new components swapped; no long video or algorithm-vs-Dense quality test'))
    lines=['# 复用旧版优化实现：短视频验收','', '保持当前 C_s17、Risk、Full_Pset_s17、Adaptive φ 和 B16 权重，512交换/no-op等算法配置不变。通过可选legacy_components开关接入旧实现，默认仍为空，未覆盖已接受基线。', '', '测试为单个city_walk_s701短视频：9 chunks，105帧（6.5625秒），768×512，5个历史chunks（360页），当前72页。每组完整预热后测两轮；另跑一次阶段profile。生成＋VAE为各自预热后阶段中位数之和，不包含视频编码/指标计算；不是上一轮30秒实验的整体计时口径。', '', '## 模块隔离', '', '|模块|当前/ms|旧实现/ms|加速|输出NMSE|', '|---|---:|---:|---:|---:|']
    for n,r in micro.items():lines.append(f'|{n}|{r["before_ms"]:.4f}|{r["after_ms"]:.4f}|{r["speedup"]:.3f}×|{r["nmse"]:.3g}|')
    lines+=['', 'RoPE使用FP64融合核，逐位等价。Risk复用旧normalizer/risk两核，直接消费现有M16 QK logits，不重复QK；仅为torch.compile显式传BP并将sqrt(D)写成等值常量表达式。Attention直接读取常驻KV和最终LUT，复用旧Page64精算核＋完整V均值/rank8补偿及LSE合并。摘要复用旧three_term_features特征投影。Reader、Adaptive φ、集合评分与动作接受逻辑未修改。', '', 'Attention主表使用当前432页/72tiles形状，捕获的单头/单tile数据扩展用于隔离测速；410页首轮结果保留在attention/，不混用为最终速度。Risk用真实捕获的432页路由输入；摘要是72个当前页。每个模块保存CUDA profile；Attention充分预热两组后再测速。', '', '理论工作量：旧补偿先将R4概率聚合成每页均值权重，均值AV从4个128维代表降为1个完整128维均值，再加3×rank8细节。考虑现有四乘积与旧两乘积，补偿AV发射乘加量比例为4×4×128 / [2×(128+3×8)] = 6.737×；精算/QK等不变，所以整个Attention收益小于此比例。当前432页补偿AV约97.84 GFLOPs，旧版约14.52 GFLOPs（不含最终低秩重建、softmax、mask等）。RoPE每视频调用29×9×5=1305次，模块差约0.768ms，预测节省约1秒，与端到端实测对应。', '', '## 完整短视频', '', '主对比（当前基线、四项全部替换、Dense）均在GPU5测量，RoPE单项与保留当前摘要组合分别在GPU7/6作筛查；这些次要组合的小幅速度差不用于精细排序。baseline_samegpu与初次baseline latent逐位一致；阶段profile取初次完整instrumented运行。', '', '|实现|生成/s|VAE/s|合计/s|生成相对基线|合计相对基线|', '|---|---:|---:|---:|---:|---:|']
    labels={'baseline':'当前基线','rope_only':'仅旧RoPE','all':'旧RoPE＋Risk＋Attention＋摘要','no_summary':'旧RoPE＋Risk＋Attention','dense':'Dense（速度参考）'}
    for n,r in rows.items():lines.append(f'|{labels[n]}|{r["generation_seconds"]:.3f}|{r["vae_seconds"]:.3f}|{r["total_seconds"]:.3f}|{r["generation_speedup_vs_baseline"]:.3f}×|{r["total_speedup_vs_baseline"]:.3f}×|')
    lines+=['', '## 替换实现引入的误差', '', '以下仅对比当前基线；PSNR/SSIM/LPIPS窗口为[21,101)，PSNR原分辨率，LPIPS/SSIM为256×384。RGB差统计全部105帧。', '', '|实现|latent NMSE|PSNR|SSIM|LPIPS|RGB MAE（0–255）|RGB最大差（0–255）|', '|---|---:|---:|---:|---:|---:|---:|']
    for n,r in errors.items():lines.append(f'|{labels[n]}|{r["latent_nmse"]:.6g}|{r["psnr"]:.3f}|{r["ssim"]:.4f}|{r["lpips"]:.4f}|{r["rgb_mae_255"]:.3f}|{r["rgb_max_abs_255"]:.0f}|')
    lines+=['', 'RoPE是数学及实测逐位等价替换。其他模块的差异来自归约顺序、投影/重建顺序和已有BF16分解数值路径，算法表示与完整V均值保留；局部微小差异可能改变Risk边界、离散换页和闭环轨迹。不能把这些候选当作无误差实现；完整数值记录保留，不据本次短视频宣称长视频质量验收。', '', '## 阶段profile', '', '|实现|Attention/s|路由及修正/s|摘要及编码/s|clean commit/s|', '|---|---:|---:|---:|---:|']
    for n in names:
        st=rows[n]['stages'];vals=[st[k]['milliseconds']/1000 for k in ['attention','route_and_correction','summary_and_encoding','clean_commit']];lines.append('|'+labels[n]+'|'+'|'.join(f'{v:.3f}' for v in vals)+'|')
    lines+=['', '模块profile、输入输出规约、逐轮耗时与源码快照均保存在本目录。加载改为使用/tmp/streaming_sparse-0/runtime中已有的本地大文件，checkpoint SHA仍校验；加载/首次编译均不计时。初次接入的编译兼容和本地路径失败尝试单独留档，未纳入耗时。', '', '实现：legacy_components.py为旧kernel适配；runtime.py增加可选组件开关；benchmark_hotspots.py增加实验参数与本地资源选择。生产默认行为未切换。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(performance=rows,errors=errors),indent=2))
if __name__=='__main__':main()
