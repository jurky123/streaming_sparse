"""Shared page conversion: cost, numerical attribution and video acceptance."""
import json
import torch,numpy as np
from .weights import ROOT

def main():
    torch.set_num_threads(4);root=ROOT/'artifacts/video_runtime_v1/reader_page32_v1';rows={}
    for n in ['baseline','page32']:
        x=json.loads((root/n/'result.json').read_text());v=json.loads((root/n/'decode.json').read_text())
        rows[n]=dict(generation=x['median_seconds'],trials=x['seconds'],vae=v['median_seconds'],total=v['generation_plus_decode_seconds'],stages=x['stages'])
    a=torch.load(root/'baseline/latents.pt',weights_only=True).float();b=torch.load(root/'page32/latents.pt',weights_only=True).float()
    x=np.load(root/'baseline/pixels.npy',mmap_mode='r');y=np.load(root/'page32/pixels.npy',mmap_mode='r')
    error=dict(latent_equal=torch.equal(a,b),latent_nmse=float((b-a).square().sum()/a.square().sum()),rgb_equal=bool(np.array_equal(x,y)),rgb_mae_255=float(sum(np.abs(i.astype(np.float64)-j).sum() for i,j in zip(x,y))/x.size))
    if (root/'quality.json').exists():
        q=json.loads((root/'quality.json').read_text());error.update({k:q[k] for k in ['psnr','ssim','lpips','window']})
    rt=json.loads((root/'full_route_steady/result.json').read_text());times={k:v['gpu_ms_median'] for k,v in rt['rows'].items()}
    out=dict(rows=rows,error=error,route_ms=times,generation_speedup=rows['baseline']['generation']/rows['page32']['generation'],total_speedup=rows['baseline']['total']/rows['page32']['total'],default_changed=False)
    (root/'acceptance.json').write_text(json.dumps(out,indent=2))
    lines=['# Reader共享投影转换：本轮验收','','基线为已逐位验收的A-only＋Reader两阶段＋旧RoPE，lazy回退关闭。权重、预算、候选、log质量、动作阈值均不变。将kernel逐Query读取FP64 mean/detail再转FP32，改为forward内先转换一次并共享。没有新增截断、低秩近似或低精度算术；但编译器的归约布局发生变化，结果不保证逐位一致。','','完整筛查见../reader_vector_v1/REPORT.zh.md。向量拆分、寄存器上限、响应系数打包均未带来收益；仅共享转换在216/288/360/432页评分中有1.049–1.067×收益，包含转换成本。','','理论：页面投影由64个FP64转为64个FP32，H12/P432占用从2.654MB降到1.327MB；72×16 Query复用同次转换。逻辑重复加载约少3.06GB，但大部分可能命中L2，不能把它当作HBM节省。向量kernel寄存器255→167；283MB的标量工作区不变。','','数值定位：同输入候选端点center/rp和log状态逐位一致，仅页面加权汇总state不同，最大绝对差9.54e-7。因此固定输入差异来自归约计算顺序；视频差异是否由哪个离散动作放大尚未逐事件定位。','','## 完整路由与视频','','首次完整路由测得5.290→6.145ms，同时其他未修改kernel也变慢；保留在full_route_profile。再次测量并在两张已捕获图间交替预热，稳态结果如下，不把首次异常结果覆盖或混入平均。',f'完整路由：{times["before"]:.3f}→{times["after"]:.3f}ms，{times["before"]/times["after"]:.3f}×。','','GPU3同卡顺序单视频，9 chunks/105帧、768×512、360历史页；完整预热后两轮生成，再单独profile/VAE。总耗时为分别预热后的生成＋VAE，不含编译、加载、编码。','','|实现|生成/s|含VAE/s|','|---|---:|---:|']
    for n,r in rows.items():lines.append(f'|{n}|{r["generation"]:.3f}|{r["total"]:.3f}|')
    lines += ['',f'生成加速{out["generation_speedup"]:.3f}×，含VAE加速{out["total_speedup"]:.3f}×。数值：{error}。','','可选--reader-page32，默认关闭。没有运行15秒/30秒。代码增加Reader向量形状/寄存器/打包实验选项，默认仍原值；诊断与短视频入口均在local_experiments/video_runtime_v1。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
