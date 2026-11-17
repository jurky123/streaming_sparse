"""Record short-video acceptance and the remaining route cost envelope."""
import json
from pathlib import Path
import torch,numpy as np
from .weights import ROOT

def main():
    torch.set_num_threads(4);root=ROOT/'artifacts/video_runtime_v1/lazy_video_v1'
    rows={}
    for name in ['baseline','lazy']:
        d=json.loads((root/name/'result.json').read_text());v=json.loads((root/name/'decode.json').read_text())
        rows[name]=dict(generation=d['median_seconds'],total=v['generation_plus_decode_seconds'],vae=v['median_seconds'],stages=d['stages'],trials=d['seconds'])
    z=torch.load(root/'baseline/latents.pt',weights_only=True).float();w=torch.load(root/'lazy/latents.pt',weights_only=True).float()
    a=np.load(root/'baseline/pixels.npy',mmap_mode='r');b=np.load(root/'lazy/pixels.npy',mmap_mode='r')
    errors=dict(latent_equal=torch.equal(z,w),latent_nmse=float((w-z).square().sum()/z.square().sum()),rgb_equal=bool(np.array_equal(a,b)))
    if (root/'quality.json').exists():
        quality=json.loads((root/'quality.json').read_text());errors.update({k:quality[k] for k in ('psnr','ssim','lpips','window')})
    errors['rgb_mae_255']=float(sum(np.abs(x.astype(np.float64)-y).sum() for x,y in zip(a,b))/a.size)
    modules={}
    for p in (ROOT/'artifacts/video_runtime_v1/lazy_fallback_v1').glob('*/result.json'):
        d=json.loads(p.read_text());v=d['rows']['layout'];modules[p.parent.name]=dict(before=d['rows']['baseline']['steady']['gpu_ms_median'],after=v['steady']['gpu_ms_median'],speedup=d['speedup'],max_abs=v['max_abs'],actions_equal=v['actions_equal'],noop_equal=v['noop_equal'])
    route=json.loads((root/'full_route_profile/result.json').read_text());rt={k:v['gpu_ms_median'] for k,v in route['rows'].items()}
    result=dict(rows=rows,errors=errors,modules=modules,route_ms=rt,generation_speedup=rows['baseline']['generation']/rows['lazy']['generation'],total_speedup=rows['baseline']['total']/rows['lazy']['total'])
    (root/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    lines=['# 按需回退评分：短视频验收','','保持C_s17、Full_Pset_s17、Adaptive φ、B16和10%预算、512交换＋no-op不变。基线为A-only＋Reader两阶段＋逐位等价旧RoPE；仅将两个每次都执行的排除候选向量聚合，改为数值不稳定动作触发时重建。log质量、正数组归一化及g>0规则保持不变。','','理论：H12/T72/M16、R16/A32、D16下，两次FP64排除聚合矩阵乘共2×13824×(16²＋32²)×16＝0.566 GFLOPs；相关完整归一化权重共141.56 MB。正常动作不需要这些向量，移除该计算和中间量；触发回退时用正质量重建，不作主导质量直接相减。重建求和次序改变，属于浮点归约差异，不是截断/降阶近似。','','## 模块验收','','|形状/组合|前/ms|后/ms|加速|最大评分差|动作/no-op|','|---|---:|---:|---:|---:|---|']
    for k,d in sorted(modules.items()):lines.append(f'|{k}|{d["before"]:.3f}|{d["after"]:.3f}|{d["speedup"]:.3f}×|{d["max_abs"]:.3g}|{d["actions_equal"]}/{d["noop_equal"]}|')
    lines+=['','split后缀比较A-only与A-only＋split＋lazy，其余只比较lazy开关。已有主导页移出/微小正分稳定性测试通过，见reader_split_v1/lazy_diagnostic。',f'整条路由以真实保存输入、原生Query布局预热216→288→360→432页：{rt["before"]:.3f}→{rt["after"]:.3f} ms，{rt["before"]/rt["after"]:.3f}×。','','## 单视频','','city_walk_s701、9 chunks/105帧、768×512、16FPS、360历史页。GPU3顺序运行，每组完整预热后测两轮，再独立profile；VAE单独预热计时，合计不含编译/加载/视频编码。','','|版本|生成/s|含VAE/s|','|---|---:|---:|']
    for n,d in rows.items():lines.append(f'|{n}|{d["generation"]:.3f}|{d["total"]:.3f}|')
    lines += ['',f'生成加速{result["generation_speedup"]:.3f}×，含VAE加速{result["total_speedup"]:.3f}×。数值验收：{errors}。','', '实现：benchmark_hotspots.py新增--lazy-scores；run_lazy_video.py成对测试；profile_lazy_route.py隔离完整路由；评分实现位于scores_packed.py、scores_moments.py和reader_reduce.py。默认LAZY仍为false。未运行15秒或30秒视频。','', '剩余热点：Reader向量阶段1.446ms，Risk约0.660ms，标量动作评分0.337ms。Reader增大warps到8/16已验证未改善且改变动作，不能简单以减少寄存器数量判断优化成功。下一步应检查向量系数重复读取和归约布局，同时保留已验证的动作语义。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
