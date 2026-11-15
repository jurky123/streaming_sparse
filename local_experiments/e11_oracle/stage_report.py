import json,csv,statistics,hashlib
from pathlib import Path
ROOT=Path('/tmp/e11_results')
def main():
 def rows(name):return list(csv.DictReader((ROOT/name).open()))
 local=rows('L1.csv');cert=rows('full_P_certificates.csv');d02=rows('D02.csv');l2=rows('L2.csv');l3=rows('L3.csv')
 lines=['E11 阶段结论：已完成全页 pilot 与在线路由闭环对照','',
 '冻结基线 178f95a8 / summary_pretrained。原始 BF16 QKV、实际缓存摘要、固定10%预算。',
 '本文件只汇总已完成数据；O-OPT 闭环尚未完成，不宣称任务全部完成。','',
 '1. 独立路由的当前表现（12 个预登记 head–tile，宏平均 NMSE）']
 for method in ('O-MASS','O-RISK','O-OPT'):
  a=[x for x in local if x['method']==method];lines.append(f"   {method}: {statistics.mean(float(x['nmse']) for x in a):.8f}; post-W_O 单 head 替换 NMSE {statistics.mean(float(x['post_WO_single_head_nmse']) for x in a):.8f}")
 lines+=['   risk 在 10/12 个实例优于 mass，宏平均局部 NMSE 降低约12.47%。',
 '   O-OPT 的12个当前集合都等于 mass/risk 中真实64-query误差较小的 warm start。',
 '   因此其相对 risk 的约0.54%均值改善不是整数搜索发现的新集合收益。','',
 '2. 全页最优性证书',
 '   GLOBAL_TOL: 0/12；BOUNDED_INCUMBENT: 3/12；NO_FEASIBLE_CERTIFICATE: 9/12。',
 '   后9例仅保留验证过的基线上界与平凡非负下界，不包装成精确优化成功。',
 '   三个非平凡界的相对间隙约84.26%、10.77%、18.69%；详见 full_P_certificates.csv。',
 '   未达到扩展门槛，不做864实例全tile扩展，也不做全覆盖L2/L3。',
 '   小问题完整枚举与全页结论严格分开。Gurobi许可内六个MIQP均与枚举一致；',
 '   SCIP较大的缩小问题含超时未闭合实例，所有失败/调试记录保留。','',
 '3. D02：不能将当前收益归因于集合抵消',
 '   当前三组的平均跨页项C为正；净抵消(kappa<1)只出现在约7%的query。',
 '   mass→risk 的NMSE改善中，按实际分母归一化的A与C均下降；',
 '   当前best-warm相对risk的微小收益来自A下降，C反而上升。',
 '   使用Dense分母的仅分析量：mass 0.00682、risk 0.00606；',
 '   对应实际变化分母NMSE为0.01950、0.01707。分母效应不可忽略，',
 '   但该分析量不是优化结果，也不能据此分离所有交互作用的因果贡献。',
 '   没有足够O-OPT证书，尚不能判断真正最优集合是否主要利用抵消。','',
 '4. 已完成闭环：所有三组约定相同采样位置，当前仅mass/risk完整结果可比较',
 '   两prompt×seed101；9chunks/105帧；RGB[21,101)。',
 '   L2五层(1/8/15/22/29)，每视频36个head–tile；risk在10个配对中latent NMSE赢5个。']
 for method in ('O-MASS','O-RISK'):
  a=[x for x in l2 if x['method']==method];lines.append(f"   L2 {method}: latent {statistics.mean(float(x['latent_nmse']) for x in a):.6f}, LPIPS {statistics.mean(float(x['lpips']) for x in a):.6f}, SSIM {statistics.mean(float(x['ssim']) for x in a):.6f}, PSNR {statistics.mean(float(x['psnr']) for x in a):.3f}")
 for x in l3:lines.append(f"   L3 {x['case']} {x['method']}: latent {float(x['latent_nmse']):.6f}, LPIPS {float(x['lpips']):.6f}, SSIM {float(x['ssim']):.6f}, PSNR {float(x['psnr']):.3f}")
 lines+=['   L3只介入五层、每视频180个head–tile，不是layer1–29全稀疏。',
 '   在所选层的完整稀疏调用中，采样比例约0.14881%；相对29层全覆盖，L3约0.02566%。',
 '   逐帧曲线保存在各视频result.json。所有候选沿自身轨迹更新缓存和mask。','',
 '5. 调整后的顺序与资源',
 '   用户确认CPU上限128核时，每实例保留900秒。',
 '   先发布本阶段报告；优先horse_field/layer15的O-OPT L2，其余L2同阶段完成；',
 '   L3放在L2之后。当前预算预留horse_field五层L2和采样L3共360次求解。',
 '   pilot消耗约3.24核时；建模P95约85.51秒，360次闭环预估约98.55核时。',
 '   另一prompt的O-OPT闭环未纳入当前计划，明确列为未完成；不会降低单实例预算补数量。','',
 '当前判断：risk仍是更合适的默认在线候选，但优势具有场景/层依赖；',
 '真正集合最优的剩余空间尚未测清，现阶段不足以据此决定投入学习集合机制。',
 '下一条有新增价值的结果是layer15闭环中的有界搜索是否改善最终输出。']
 (ROOT/'STAGE_CONCLUSIONS.zh.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
