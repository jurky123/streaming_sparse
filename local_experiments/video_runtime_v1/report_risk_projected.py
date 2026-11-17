"""Risk fusion acceptance, with shared-kernel attribution kept explicit."""
import json,hashlib
from pathlib import Path
from .weights import ROOT

def main():
    root=ROOT/'artifacts/video_runtime_v1/risk_projected_video_v1';rows={}
    for n in ['baseline','projected']:
        r=json.loads((root/n/'result.json').read_text());d=json.loads((root/n/'decode.json').read_text())
        rows[n]=dict(generation=r['median_seconds'],trials=r['seconds'],total=d['generation_plus_decode_seconds'],vae=d['median_seconds'],stages=r['stages'])
    errors=json.loads((root/'errors.json').read_text());q=json.loads((root/'quality.json').read_text());errors.update({k:q[k] for k in ['psnr','ssim','lpips','window']})
    assert errors['baseline_matches_current'] and errors['latent_equal'] and errors['rgb_equal']
    modules={}
    for p in [216,288,360,432]:
        r=json.loads((ROOT/f'artifacts/video_runtime_v1/risk_projected_v1/p{p}_b16_w8_s0f1l1_per0/result.json').read_text())
        assert r['mask_equal'] and r['remove_equal'] and r['add_equal']
        modules[p]=dict(before=r['rows']['baseline']['gpu_ms_median'],after=r['rows']['projected']['gpu_ms_median'],speedup=r['speedup'],nmse=r['nmse'],max_abs=r['max_abs'],mask_equal=True,remove_equal=True,add_equal=True)
    r=json.loads((root/'full_route/result.json').read_text());route={k:v['gpu_ms_median'] for k,v in r['rows'].items()}
    result=dict(rows=rows,modules=modules,route_ms=route,errors=errors,generation_speedup=rows['baseline']['generation']/rows['projected']['generation'],total_speedup=rows['baseline']['total']/rows['projected']['total'],canonical_baseline_adds='--projected-risk',generic_default=False)
    (root/'acceptance.json').write_text(json.dumps(result,indent=2));(root/'status.json').write_text(json.dumps(dict(status='complete_short_video_bitwise',canonical_flag='--projected-risk',next='Reader FP64 logmass repeated page reduction'),indent=2))
    lines=['# 保留原投影的Risk融合验收','','保留原FP32投影SGEMM，只融合其后的Risk特征及M16归约；完整权重、算法配置、候选规则、精算/补偿均不变。与此前旧Risk融合不同，不把8维投影改成Triton顺序累加。最终BP16/WARPS8、feature顺序相加、libdevice sqrt，未降精度。','','24组布局/算术次序/拆分M16对照后选择融合版本，再做4种窗口复验。局部Risk不是逐位一致，NMSE约3–5e-15；初选mask、移出池和加入池均一致。M16拆分没有改善误差且变慢，不采用。','','## 隔离Risk','','|页数|前/ms|后/ms|加速|NMSE|','|---:|---:|---:|---:|---:|']
    for p,r in modules.items():lines.append(f'|{p}|{r["before"]:.3f}|{r["after"]:.3f}|{r["speedup"]:.3f}×|{r["nmse"]:.3g}|')
    lines += ['', '风险数值量级较大，432页最大绝对差0.125不能脱离NMSE单独解读；所有固定输入路由池契约已核对。', '', '## 完整路由与视频', '',f'完整路由{route["before"]:.3f}→{route["after"]:.3f}ms（{route["before"]/route["after"]:.3f}×），远小于隔离Risk倍数。原编译器把Risk与Reader log质量计算融合在一个约0.660ms的kernel中，并有特征stack/mean等后处理；此前“Risk约0.875ms”的表格标签包含共享工作且不是全部Risk开销，不能当作纯Risk独占时间。本轮拆分后Reader FP64 logmass约0.560ms重新显露，Risk三项主要kernel为0.172ms归约＋0.230ms概率/查询预处理＋0.066ms投影。', '', 'GPU3同卡顺序，city_walk_s701、9chunks/105帧、768×512、16FPS、360历史页。每组完整预热后两轮计时，VAE单独预热；合计不含编译、加载、编码。两组都开旧快Attention、A-only/split Reader和scan-exclude，PAGE32/lazy关闭。', '', '|实现|生成/s|含VAE/s|','|---|---:|---:|']
    for n,r in rows.items():lines.append(f'|{n}|{r["generation"]:.3f}|{r["total"]:.3f}|')
    lines += ['',f'生成加速{result["generation_speedup"]:.4f}×，含VAE加速{result["total_speedup"]:.4f}×。latent/RGB逐位一致，RGB差0、LPIPS0、PSNR100dB（零差封顶），SSIM实现输出{errors["ssim"]:.9f}。基线latent与上一轮scan输出逐位一致。', '', '|阶段|baseline/s|projected/s|','|---|---:|---:|']
    for k in ['attention','route_and_correction','summary_and_encoding','clean_commit']:lines.append(f'|{k}|{rows["baseline"]["stages"][k]["milliseconds"]/1000:.3f}|{rows["projected"]["stages"][k]["milliseconds"]/1000:.3f}|')
    lines+=['','## 下一热点和记录','','当前Reader log质量kernel三次遍历页面，生成代码含13处exp和4处log（静态表达式位置，不是硬件指令计数），重复形成R4局部logsumexp。编译证据在compiler_evidence/；下一步检查是否值得缓存局部log质量再归一化。不能仅看算术量就预测收益，要比较完整路由中的融合得失。','','risk_projected.py为保留投影的融合实现；runtime.py/benchmark_hotspots.py新增可选projected_risk；当前命令显式加入--projected-risk，通用参考默认关闭。短视频运行脚本run_risk_projected_video.py，隔离脚本check_risk_projected.py。暂停/恢复后检查进程均已退出且结果齐全，没有重启已完成任务。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
