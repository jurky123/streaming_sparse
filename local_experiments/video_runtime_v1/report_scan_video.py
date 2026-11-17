"""Positive prefix/suffix aggregation acceptance and cost report."""
import json,hashlib
from pathlib import Path
from .weights import ROOT

def main():
    root=ROOT/'artifacts/video_runtime_v1/exclude_scan_video_v1';rows={}
    for name in ['baseline','scan']:
        r=json.loads((root/name/'result.json').read_text());d=json.loads((root/name/'decode.json').read_text())
        rows[name]=dict(generation=r['median_seconds'],trials=r['seconds'],vae=d['median_seconds'],total=d['generation_plus_decode_seconds'],stages=r['stages'])
    err=json.loads((root/'errors.json').read_text());q=json.loads((root/'quality.json').read_text());err.update({k:q[k] for k in ['psnr','ssim','lpips','window']})
    assert err['baseline_matches_current'] and err['latent_equal'] and err['rgb_equal']
    modules={}
    for pages in [216,288,360,432]:
        p=ROOT/f'artifacts/video_runtime_v1/exclude_scan_v1/kind_a_p{pages}_b32_w4_g4_bw4/result.json';d=json.loads(p.read_text());x=d['rows']['layout']
        assert x['actions_equal'] and x['noop_equal']
        modules[pages]=dict(before=d['rows']['baseline']['steady']['gpu_ms_median'],after=x['steady']['gpu_ms_median'],speedup=d['speedup'],max_abs=x['max_abs'],actions_equal=True,noop_equal=True)
    route=json.loads((root/'full_route_attributed/result.json').read_text());rt={k:v['gpu_ms_median'] for k,v in route['rows'].items()}
    result=dict(rows=rows,errors=err,modules=modules,route_ms=rt,generation_speedup=rows['baseline']['generation']/rows['scan']['generation'],total_speedup=rows['baseline']['total']/rows['scan']['total'],status='short_video_bitwise_pass',current_command_adds='--scan-exclude',generic_reference_switch_default=False)
    (root/'acceptance.json').write_text(json.dumps(result,indent=2));(root/'status.json').write_text(json.dumps(dict(status='complete_short_video_bitwise',canonical_baseline='old fast attention + A-only split Reader + scan exclusion',next='Reader vector reuse; Risk arithmetic/layout'),indent=2))
    lines=['# 正质量前后缀排除汇总：验收','','本轮将16/32候选组的逐排除softmax＋FP64 BMM改为正质量前缀/后缀扫描。没有主导质量相减，没有降精度或截断；移出最大质量页时在其余页最大值上单独重新归一化。归约顺序改变，固定输入评分存在FP64微差，但四种窗口最高动作/no-op一致，完整短视频latent/RGB逐位一致。','','## 算法和理论工作量','','对组内最大log质量缩放后，将第i项之前和之后的正质量、加权Value分别累计相加，得到排除i后的结果。唯一移出最大页的候选另用第二大log质量缩放，保证1000-log差距下剩余信息仍保留。表达式没有总质量减被移出质量。','','参考的K×K softmax矩阵不再物化；两个组共141557760字节权重，省去相关写入/读取。原两次BMM为0.566GFLOPs。扫描无需每个排除项重算全部K页；GPU并行扫描包含logK步，实际操作量按O(K D logK)理解，不能把概念上的线性前缀算法直接当作已发射线性FLOPs。','','理论距离修正：单看0.566GFLOPs/13.68TFLOP/s得到0.041ms，实测0.372ms似乎差9倍；计入weights＋values＋output最低逻辑payload约311.43MB，以1266.4GB/s得到0.246ms，距离约1.51倍。缓存会影响实际HBM流量，此处是payload参考，不是硬件计数器实测。新扫描的唯一输入/输出payload约180.49MB，对应0.143ms参考；本轮两个scan kernel合计0.293ms。','','## 四种窗口完整评分','','|页数|前/ms|后/ms|加速|最大评分差|动作/no-op|','|---:|---:|---:|---:|---:|---|']
    for p,d in modules.items():lines.append(f'|{p}|{d["before"]:.3f}|{d["after"]:.3f}|{d["speedup"]:.3f}×|{d["max_abs"]:.3g}|一致|')
    lines += ['',f'完整路由稳态{rt["before"]:.3f}→{rt["after"]:.3f}ms，{rt["before"]/rt["after"]:.3f}×。另一次独立profile为5.304→4.990ms。scan kernel重新命名以分离原QK kernel统计，归因profile保存在full_route_attributed。旧排除归约＋BMM约0.655ms，新scan约0.293ms；整个路由收益小于这一局部比值。','','## 单视频验收','','GPU3同卡顺序：city_walk_s701，9 chunks/105帧、768×512、16FPS、360历史页；各完整预热后两次计时，阶段profile和VAE独立测量，总耗时不含加载/编译/编码。精算/补偿固定用户选定的旧快实现，A-only＋split Reader、当前Risk/摘要/Full_Pset，PAGE32/lazy均关闭。','','|实现|生成/s|含VAE/s|','|---|---:|---:|']
    for n,r in rows.items():lines.append(f'|{n}|{r["generation"]:.3f}|{r["total"]:.3f}|')
    lines += ['',f'生成{result["generation_speedup"]:.3f}×，含VAE{result["total_speedup"]:.3f}×。latent/RGB逐位一致、NMSE0、RGB MAE0；PSNR记录100dB为零误差上限，LPIPS0，SSIM实现返回{err["ssim"]:.9f}（相同图像的浮点实现值）。独立基线latent与fast_attention_baseline_v1/current逐位一致。','','|模块|baseline/s|scan/s|','|---|---:|---:|']
    for k in ['attention','route_and_correction','summary_and_encoding','clean_commit']:
        lines.append(f'|{k}|{rows["baseline"]["stages"][k]["milliseconds"]/1000:.3f}|{rows["scan"]["stages"][k]["milliseconds"]/1000:.3f}|')
    lines+=['','14份E49＋主导页稳定性案例在G1/G4通过；另8个直接聚合案例覆盖K16/32与最大页在头/中/尾、1000-log质量差，均保留有限背景。相关结果在../exclude_scan_v1/stability_g4及aggregate_validation.json。','','## 未采用方案和改动','','先试8组纯FP64 BMM替换、8组softmax＋向量汇总直接融合，均变慢；再试8组扫描配置，选择GROUP4/WARPS4，四窗口复验。源码不支持tl.gather的首次编译失败留在日志，不计入结果；最终直接偏移加载并扫描，无不受支持API。','','exclude_scan.py新增正质量扫描；scores.py新增默认关闭的实验开关；benchmark_hotspots新增--scan-exclude。当前运行配置local_setup/CURRENT_VIDEO_BASELINE.txt已加入此选项；通用参考入口默认关闭以保留历史对照。下一轮优化基线应显式开启扫描。未运行15秒/30秒视频。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
