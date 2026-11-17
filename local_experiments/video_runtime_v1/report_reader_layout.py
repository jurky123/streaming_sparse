"""Record numerical and speed acceptance of the A-only coefficient layout."""
import json
from pathlib import Path
import torch,numpy as np
from .weights import ROOT
from ..learned_cache.io import atomic


def main():
    torch.set_num_threads(2);root=ROOT/'artifacts/video_runtime_v1/reader_layout_v1';names=['baseline','layout','rope_layout','legacy_layout'];rows={}
    ref=torch.load(root/'baseline/latents.pt',weights_only=True);rgb=np.load(root/'baseline/pixels.npy')
    for name in names:
        r=json.loads((root/name/'result.json').read_text());d=json.loads((root/name/'decode.json').read_text())
        if name=='legacy_layout':
            prior=ROOT/'artifacts/video_runtime_v1/legacy_reuse/all';zref=torch.load(prior/'latents.pt',weights_only=True);imref=np.load(prior/'pixels.npy');reference='legacy_reuse/all'
        else:zref=ref;imref=rgb;reference='baseline'
        z=torch.load(root/name/'latents.pt',weights_only=True);im=np.load(root/name/'pixels.npy')
        rows[name]=dict(generation_seconds=r['median_seconds'],generation_trials=r['seconds'],total_seconds=d['generation_plus_decode_seconds'],vae_seconds=d['median_seconds'],latent_equal=torch.equal(z,zref),rgb_equal=bool(np.array_equal(im,imref)),reference=reference,stages=r['stages'],flags=dict(reader_layout=r.get('reader_layout'),kind=r.get('reader_layout_kind'),legacy_components=r.get('legacy_components')))
        assert rows[name]['latent_equal'] and rows[name]['rgb_equal'],name
    for x in rows.values():x['generation_speedup']=rows['baseline']['generation_seconds']/x['generation_seconds'];x['total_speedup']=rows['baseline']['total_seconds']/x['total_seconds']
    modules={}
    for pages,path in [(216,'kind_a_p216'),(288,'kind_a_p288'),(360,'kind_a_p360'),(432,'kind_a')]:
        r=json.loads((root/path/'result.json').read_text());x=r['rows']['layout'];assert x['bitwise_equal'] and x['actions_equal'] and x['noop_equal']
        modules[pages]=dict(before_ms=r['rows']['baseline']['steady']['gpu_ms_median'],after_ms=x['steady']['gpu_ms_median'],speedup=r['speedup'],scores_equal=True,actions_equal=True,noop_equal=True)
    route=json.loads((root/'full_route_profile/result.json').read_text());route_times={k:v['gpu_ms_median'] for k,v in route['rows'].items()}
    result=dict(status='accepted_short_video_bitwise',rows=rows,modules=modules,full_route_ms=route_times,config=dict(chunks=9,frames=105,resolution=[512,768],history_pages=360,history_chunks=5,fps=16),scope='Current baseline comparison on GPU3; legacy composition on GPU4 versus prior GPU5 is a compatibility check, not same-device fine timing.',next_hotspot='Reader remains ~2.29ms / 40% of full route; 255 registers/thread, unchanged arithmetic; investigate register pressure and repeated scalar/vector phase live ranges.')
    atomic(root/'acceptance.json',result)
    labels={'baseline':'原当前基线','layout':'A-only Reader布局','rope_layout':'A-only＋逐位等价RoPE','legacy_layout':'A-only＋旧算子候选组合'}
    lines=['# Reader页面系数布局优化验收','', '本轮只改变Adaptive φ页面系数A的存储布局，不改变网络、数据精度、运算公式、Risk或Full_Pset动作规则。四个历史窗口的评分、argmax和no-op逐位一致；9 chunks完整短视频的latent与RGB逐位一致。', '', '## 定位与理论依据','', '此前Reader是路由热点。A包含每页8×4个FP32系数，原布局[H,P,8,4]使固定系数跨页面访问的地址间隔为128字节；转为[H,8,4,P]后为4字节。432页、12heads仅有663552字节A数据，转置开销包含在测试里；每个forward只打包一次，供72个Query tiles共享。算术量完全不变，8维×4输出投影在H12/T72/M16/P432上的乘加量约0.382 GFLOPs；实测提速来自访存布局改善。该投影FLOPs不包含tanh、响应、归约及评分，不能作为整个Reader的完整理论下限。', '', '同时尝试了拆分CO响应/β布局，寄存器数量下降但没有提速，并出现约9.21e-9的评分差及动作/no-op变化。CO-only、A＋CO两组留档，不采用。最终仅启用A-only，CO与所有FP64页面投影保持原状。', '', '## 模块隔离（含转置及完整集合评分）','', '|页数|优化前/ms|优化后/ms|加速|评分/动作/no-op|','|---:|---:|---:|---:|---|']
    for p,r in modules.items():lines.append(f'|{p}|{r["before_ms"]:.3f}|{r["after_ms"]:.3f}|{r["speedup"]:.3f}×|逐位一致|')
    lines+=['',f'采用真实捕获路由输入；216/288/360页测试截取页面并重建对应own索引。整条路由使用原生Query步幅、逐步预热216→288→360→432页后测量，从{route_times["before"]:.3f}降至{route_times["after"]:.3f}ms，{route_times["before"]/route_times["after"]:.3f}×。profile见full_route_profile/。主计时在双方充分预热后记录；初次profile采样受GPU频率爬升影响，不用它取代稳态计时。', '', '## 单视频验收','', 'city_walk_s701，768×512，9 chunks / 105帧（6.5625秒），5历史chunks / 360页，16FPS。GPU3顺序测当前基线、A-only、A-only＋RoPE，各完整预热后测两轮，另跑instrumented阶段profile。生成＋VAE为分别预热后的阶段中位数之和；不含编码、文件IO、模型加载及编译。大模型使用/tmp中的本地副本并保留SHA校验。', '', '|实现|生成/s|含VAE/s|生成相对原基线加速|含VAE加速|数值对照|','|---|---:|---:|---:|---:|---|']
    for n,r in rows.items():lines.append(f'|{labels[n]}|{r["generation_seconds"]:.3f}|{r["total_seconds"]:.3f}|{r["generation_speedup"]:.3f}×|{r["total_speedup"]:.3f}×|对{r["reference"]} latent/RGB逐位一致|')
    lines+=['', '最后一行叠加了上一轮有数值漂移的旧算子候选，它只证明A-only没有再引入误差；不代表该组合相对原当前基线无误差。其跨GPU速度仅供组合观察，主验收用GPU3同卡对照。', '', '## 改动与使用','', '- reader_layout.py：复用原Reader算术，增加A/CO布局编译期开关；验收配置仅PACK_A=true、PACK_CO=false。','- reader_reduce.py：增加A转置打包，默认LAYOUT仍为false，启用时LAYOUT_KIND默认a。','- benchmark_hotspots.py：新增--reader-layout与--reader-layout-kind；通过--reader-layout启用本轮已验收配置。','- 可与--legacy-components rope组合，二者在完整短视频上对当前基线保持逐位一致。', '', '## 优化后profile与下一步','', 'A-only后Reader仍约2.29ms，占完整路由约40%；registers/thread仍为255、无spill。其余主要成本为Risk约0.67ms、矩阵乘约0.60ms、动作归约约0.37ms。下一步重点检查Reader标量权重阶段和16维向量聚合同时存活的寄存器压力；先隔离测试，再做短视频验收。本轮未改变精度和接受阈值，也未运行15秒或30秒视频。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
