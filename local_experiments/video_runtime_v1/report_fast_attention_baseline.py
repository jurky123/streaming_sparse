"""Document the user-selected old exact/tail implementation as the new baseline."""
import json,os
from pathlib import Path
import numpy as np,torch
from .weights import ROOT
from ..high_sparse_experiment import prefix_quality

def main():
    torch.set_num_threads(4);root=ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1';rows={}
    for name in ['reference','current','dense']:
        r=json.loads((root/name/'result.json').read_text());d=json.loads((root/name/'decode.json').read_text())
        rows[name]=dict(generation=r['median_seconds'],trials=r['seconds'],vae=d['median_seconds'],total=d['generation_plus_decode_seconds'],stages=r['stages'],legacy_components=r['legacy_components'])
    z=torch.load(root/'reference/latents.pt',weights_only=True).float();w=torch.load(root/'current/latents.pt',weights_only=True).float()
    a=np.load(root/'reference/pixels.npy',mmap_mode='r');b=np.load(root/'current/pixels.npy',mmap_mode='r')
    os.environ['TORCH_HOME']=str(ROOT/'artifacts/rollout_acceptance_20261030/torch_cache')
    import lpips
    metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
    q=prefix_quality([b],[a],metric)
    error=dict(latent_equal=torch.equal(z,w),latent_nmse=float((w-z).square().sum()/z.square().sum()),rgb_equal=bool(np.array_equal(a,b)),rgb_mae_255=float(sum(np.abs(x.astype(np.float64)-y).sum() for x,y in zip(a,b))/a.size),**{k:q[k] for k in ['psnr','ssim','lpips','window']})
    for n,r in rows.items():
        r['generation_speedup']=rows['reference']['generation']/r['generation'];r['total_speedup']=rows['reference']['total']/r['total'];r['vs_dense']=rows['dense']['total']/r['total']
    result=dict(rows=rows,error_vs_previous=error,authorization='精算和补偿用旧的快实现。',default_attention='prior_fast',scope='Only exact and compensation change; current Risk, summary weights, Reader and Full_Pset preserved; A-only/split/rope shared; PAGE32 and lazy off.')
    (root/'acceptance.json').write_text(json.dumps(result,indent=2));(root/'quality.json').write_text(json.dumps(q,indent=2))
    lines=['# 当前基线：旧版快速精算与补偿','','按用户明确要求，精算和补偿使用旧快速实现，作为后续优化基线。保留当前Risk、摘要权重、Full_Pset、Adaptive φ与B16；本轮不是四项旧模块全部替换。Reader使用已逐位验收A-only＋两阶段，RoPE使用已逐位验收旧融合核；PAGE32与lazy均关闭。','','精算通过Page64直接LUT读取常驻KV，补偿为完整128维V均值＋rank8细节，旧版BF16两项AV，按base2 LSE合并。没有用低秩均值替代完整均值。该实现相对先前四乘积/代表向量实现有数值差异，用户已指定采用；报告如实保留对照。','','## 速度','','GPU3同卡顺序；单city_walk_s701，9 chunks/105帧，768×512，16FPS，360历史页。各组完整预热后两次计时，再单独阶段profile/VAE；总耗时为生成＋VAE中位数之和，不含编译/加载/视频编码。','','|实现|生成/s|含VAE/s|含VAE相对旧基线|相对Dense|','|---|---:|---:|---:|---:|']
    for n,r in rows.items():lines.append(f'|{n}|{r["generation"]:.3f}|{r["total"]:.3f}|{r["total_speedup"]:.3f}×|{r["vs_dense"]:.3f}×|')
    lines+=['','## 实现切换误差（current相对reference）','',str(error),'','PSNR/SSIM/LPIPS窗口[21,101)，PSNR原分辨率，SSIM/LPIPS为256×384；RGB MAE覆盖105帧。此处没有计算算法相对Dense的质量误差。','','## 阶段profile','','|模块|reference/s|current/s|','|---|---:|---:|']
    for key in ['attention','route_and_correction','summary_and_encoding','clean_commit']:
        lines.append(f'|{key}|{rows["reference"]["stages"][key]["milliseconds"]/1000:.3f}|{rows["current"]["stages"][key]["milliseconds"]/1000:.3f}|')
    lines+=['','## 默认入口与复现','','VideoRuntime的legacy_components缺省值为None，解析为attention；显式字符串仍按指定组件执行，空字符串可恢复旧控制。benchmark_hotspots参数缺省为attention；完整当前基线用--reader-layout --reader-split --legacy-components rope,attention。命令见local_setup/CURRENT_VIDEO_BASELINE.txt，成对验收脚本run_fast_attention_baseline.py。旧实验产物不修改。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
