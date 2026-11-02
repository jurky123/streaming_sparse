"""Read-only aggregation of high-sparsity experiments; early RGB remains primary."""
import json,math
from pathlib import Path
ROOT=Path('artifacts/high_sparse_20261030');PROBE=Path('artifacts/high_sparse_probe4_20261030')
def read(p):return json.loads(p.read_text())
def main():
    primary=read(ROOT/'early_summary.json');probe=read(PROBE/'early_summary.json')
    oracle=read(ROOT/'oracle_v/summary.json');rows=[]
    lines=['# 80–90%以上稀疏度：第一轮实验结果','','已实现可变精确页预算、历史与当前页联合选页、未选当前页补偿；原稳定路径未修改。精确稀疏度达到目标，但第一轮冻结权重实验没有达到保质验收。不能把补偿在局部attention上的改进直接当成闭环视频质量改进。','','## 前段质量','','RGB[21,101)，同已有两个prompt与seed1001；不是盲测。所有query仍计算，稀疏率包含当前KV，但仅针对稀疏调用；层0/clean/早期dense不变。','','| 路由 | 场景 | 配置 | 稳态精确稀疏率 | LPIPS↓ | SSIM↑ | PSNR↑ |','|---|---|---|---:|---:|---:|---:|']
    for router,summary in [('mean_var',primary),('probe4',probe)]:
        for r in summary['rows']:
            q=r['quality'];rate=r['runtime']['budget_trace']['338']['exact_sparsity'];case=r['case']
            old=read(Path('artifacts/compensation_probe_20261030/acceptance')/(case+'_s1001')/'mass_var.quality.json')['per_frame']
            baseline=dict(lpips=sum(old['lpips'][21:101])/80,ssim=sum(old['ssim'][21:101])/80,psnr=-10*math.log10(sum(old['mse'][21:101])/80))
            passed=q['lpips']<=baseline['lpips']+.005 and q['ssim']>=baseline['ssim']-.005 and q['psnr']>=baseline['psnr']-.2
            rows.append(dict(router=router,case=case,method=r['name'],sparsity=rate,quality={k:v for k,v in q.items() if k!='per_frame'},quality_gate_vs_original_sparse=passed))
            lines.append(f"| {router} | {case} | {r['name']} | {rate:.2%} | {q['lpips']:.5f} | {q['ssim']:.5f} | {q['psnr']:.3f} |")
    lines+=['','## 路由与V表达诊断','','global90产生的实际闭环QKV上，3层×3chunk×2step×2场景=36事件，每事件32query×12heads。下面是逐query NMSE的事件均值，不是视频LPIPS。两次诊断重放相同轨迹，不算72个独立事件；teacher使用实际缓存中的dense attention，不是新生成的dense视频轨迹。','','| 诊断 | 平均局部NMSE |','|---|---:|']
    for key,val in oracle['means'].items():lines.append(f'| {key} | {val:.6f} |')
    lines+=['','真实页质量oracle路由与残差单次swap只用于诊断，不是线上方法或全局最优上限。只换真实mass、仍用原rank8 V反而更差，再次说明分母与向量贡献需要一起校准。恢复当前页full mean V有帮助，但不能完全补足质量；all full mean V不满足现有辅助缓存约束，不能直接部署。当前页full V可作为临时方案，但仍需新kernel和计时。','','## 完整30秒计时','','含VAE，排除加载/预热/指标/编码；质量验收已失败的候选只用于计算收益诊断，不因速度通过而晋级。','','| 场景 | 方法 | 生成秒 | 含VAE秒 | 同卡无外部进程采样 |','|---|---|---:|---:|---|']
    full=[]
    for case in ['basketball_court','butterfly_garden']:
        for p in sorted((ROOT/'full'/case).glob('*.json')):
            r=read(p)
            if p.name=='stable_control.json':t=r;method='original_stable_compensation'
            else:t=r['performance'];method=r['name']
            row=dict(case=case,method=method,generation_seconds=t['generation_seconds'],seconds=t['total_generation_vae_seconds'],clean=t['telemetry']['no_observed_foreign_processes']);full.append(row)
            lines.append(f"| {case} | {method} | {row['generation_seconds']:.3f} | {row['seconds']:.3f} | {row['clean']} |")
    lines+=['','## 工程与下一步','','旧预算控制在两场景9chunk上逐位一致；16组可变LUT/补偿公式测试通过；完整轨迹若已完成，额外检查前9chunk latent与前段实验一致。常驻self-KV+辅助维持5,095,292,576字节，当前摘要是临时工作区，其计算和峰值必须计时。','','本轮结果支持下一步针对高稀疏度重新训练页路由与V子空间/聚合，尤其包含当前chunk与更大未选质量的训练分布；还应测试同总页预算下按query/head分配，以及精确替换摘要的残差收益路由。probe4是可部署的诊断性改进，尚未保质，且其FP32 eager开销没有被忽略。不要继续直接把稀疏率推高来期待固定补偿自然变好。','','文件按完整单元原子提交；主实验1.5GiB限额，独立probe4目录同样有限额，当前合计应另行核查。报告不把已有两个prompt的选择结果当成新场景确认。']
    (ROOT/'report.zh.md').write_text('\n'.join(lines)+'\n')
    result=dict(early_units=len(rows),early_expected=30,high_sparsity_quality_passed=any(r['sparsity']>=.8-1e-8 and r['quality_gate_vs_original_sparse'] for r in rows),rows=rows,full_rows=full,oracle=oracle['means'])
    (ROOT/'combined_summary.json').write_text(json.dumps(result,indent=2));print(dict(early_units=len(rows),full_units=len(full),high_sparsity_quality_passed=result['high_sparsity_quality_passed']))
if __name__=='__main__':main()
