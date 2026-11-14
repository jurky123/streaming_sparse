import json,csv,platform,shutil,tarfile
from pathlib import Path
import numpy as np
import scipy,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .stage34 import ROOT,OUT
from .common import dump,sha

def main():
 audit=json.loads((OUT/'audit.json').read_text());assert audit['passed'];selected=json.loads((OUT/'selected_model.json').read_text());t3=json.loads((OUT/'t3_summary.json').read_text());t4=json.loads((OUT/'t4_summary.json').read_text());actions=json.loads((OUT/'t3_first_action_summary.json').read_text());diag=json.loads((OUT/'diagnostics_summary.json').read_text());gpu=json.loads((OUT/'gpu_cost.json').read_text());train=json.loads((OUT/'training.json').read_text())
 def row(m,step=4,sp='evaluation',data=t3):return next(r for r in data if r['method']==m and r['step']==step and r['split']==sp)
 full=row('full_q64_full');sel=row(selected['method'],selected['steps'],data=t4)
 lines=['set_oracle_v1｜T1–T4 汇总','',
 '结论：本两场景范围内，集合交换有稳定的离线收益；Medium候选池和16维真实残差各自保留大部分四轮搜索收益。但本轮最小线性预测器未获得稳定正收益，不能把oracle表示能力等同于已实现的在线Router。保留当前Risk，不部署此预测器。','',
 '实验范围与隔离',
 '178f95a8 / summary_pretrained / G-KVR；Q64、Page64、R4/rank8、512×768，全部原模型参数冻结。city_walk、horse_field × seed101/303 × chunk2/5/8 × layer1/15/29 × step0/2 × 两个tile，共144个head–tile、4条基础轨迹。chunk2 P216/K21；chunk5/8 P410/K41。每个集合own保护。',
 'T3/T4拆分：city_walk seed101的36例拟合PCA/预测器；city_walk seed303的36例选择超参数；horse_field两个seed的72例作评价。场景曾参与开发，并非最终全新测试集；多个tile来自同一轨迹，不能按独立视频解释。全部误差为同状态pre-W_O、64个Query真实混合NMSE；不含下游视频结论。','',
 'T1｜不可变数据集',
 '144/144贡献已冻结；保存完整FP64有符号残差、分母修正、Q/摘要/风险特征、token身份及哈希。原始直接混合目标与残差重建最大差异6.44e-15。6例旧贡献通过身份核验复用；两个horse轨迹与旧基线latent逐位一致。新实验未重跑骨干或修改原始贡献。','',
 'T2｜完整集合搜索与机制',
 'Risk＋Exchange对每例较强Mass/Risk：141/144改善，3例持平；改善均值39.44%、中位36.92%。前1/2/4次交换平均获得自身最终局部收益38.19%/58.92%/80.33%。三起点最终mask相同106/144，19例Risk局部最优被另一起点超过。',
 '固定Risk分母的加性归因中，111例最大正向贡献来自交叉项，30例来自独立能量；分母变化98例帮助、39例拖累。交叉项减少不一定表示净负抵消。12个新预登记全局求解：2 GLOBAL_TOL、10 BOUNDED_INCUMBENT；不将局部最好解称全局最优。本轮T3/T4未重复求证。','',
 'T3｜候选池：horse_field 72例，真实残差、M64、最多四轮',
 '方法 | 相对Risk平均NMSE降低 | 中位降低 | 中位恢复完整四轮收益 | 首轮最佳动作召回（70个正收益实例）']
 for m in ['full_q64_full','full_q64_small','full_q64_medium','full_q64_large']:
  r=row(m);a=next(a for a in actions if a['split']=='evaluation' and a['method']==m)
  lines.append(f"{m} | {r['mean_gain_pct']:.2f}% | {r['median_gain_pct']:.2f}% | {100*r['median_full4_recovery']:.2f}% | {100*a['best_in_pool_rate']:.2f}%")
 lines+=['Small=8×16，Medium=16×32，Large=32×64；稳态Full=40×369。早期K21时移出最多20页，Large=20×64、Full=20×195，不能把所有实例成本都写成14760。候选池按各自当前集合重算，无真实标签参与候选生成。',
 'Medium的最佳动作召回仅50%，但四轮收益恢复中位94.84%；说明不必每次命中唯一最佳动作。这里仍用真实残差选池内动作，属于oracle上限。','',
 'T3｜方向表示：公共投影、全部候选、M64、四轮',
 '方法 | 平均NMSE降低 | 中位四轮收益恢复 | 比Risk更差实例/72']
 for kind in ['random','pca']:
  for rank in [4,8,16,32]:
   r=row(f'{kind}_r{rank}_q64');lines.append(f"{kind} rank{rank} | {r['mean_gain_pct']:.2f}% | {100*r['median_full4_recovery']:.2f}% | {r['worse']}/72")
 lines+=['PCA具体为仅训练拆分拟合的未中心化二阶矩主方向/SVD；不减去逐页均值，保持有符号残差可加。随机投影seed17，所有页同一空间。所有最终动作均用原始128维、64Query目标评价。16维表示有信息但并非无损，32维也仍有少量退化；平均重建误差不是这里的成功标准。',
 'M16信息对照：Full residual四轮平均改善25.76%，M64为31.31%；PCA16/M16为21.77%，M64为27.28%。实际Q64 tile未改变，只减少决策使用的probe。前20动作重叠和排序相关性见diagnostics逐实例记录。','',
 'T4｜最小预测器：信息可在线计算，结果为负',
 f"线性ridge，{train['features']}输入、16维有符号残差＋1维log页质量比，{train['features']*17}个拟合系数；训练样本行数{train['samples']}（不是独立事件数）。输入仅Q、K/V摘要、已有风险特征、年龄及全摘要输出等；Dense输出、真实残差、真实分母修正和精确页logZ仅为监督/评价标签。",
 '监督残差按每Query全摘要分母归一化；预测log精确/摘要质量比再还原正质量，确保候选分母为正。预测log比数值保护区间[-20,20]，计数逐实例记录；不裁剪任何真实贡献或认证系数。',
 f"仅开发集选择的最佳预测器：{selected['method']}，{selected['steps']}轮。比较3个正则强度、M16/M64、三档池、1/2/4轮，共54个组合。开发集平均改善{selected['validation_mean_gain_pct']:.2f}%仍为负，因此部署选择应为不修正的Risk；没有利用评价集重调参数。",
 '相同Small候选池、M64、一次交换，horse_field 72例：',
 '方法 | 相对Risk平均NMSE降低 | 变差实例/72',
 'Risk不修正 | 0.00% | 0/72']
 for label,r in [('128维真实残差',row('full_q64_small',1)),('16维真实PCA残差',row('pca_r16_q64_small',1)),('预测16维残差＋预测质量比',sel)]:lines.append(f"{label} | {r['mean_gain_pct']:.2f}% | {r['worse']}/72")
 d=diag['evaluation'];lines += [
 f"失败诊断：当前集合剩余残差方向cosine中位{d['median_set_residual_cosine']:.3f}；混合分母相对MAE均值{100*d['mean_denominator_relative_mae']:.1f}%（不等于对数误差）。单独给真实残差、保留预测分母，平均改善{d['oracle_interventions']['oracle_residual_predicted_denominator']:.2f}%；给真实分母、保留预测残差，改善{d['oracle_interventions']['predicted_residual_oracle_denominator']:.2f}%。这两个是事后离线教师干预，不能部署、未用于选择模型。方向和分母预测均有问题，不能把失败归咎于候选池容量。",'',
 'T4｜实际CUDA成本',
 'A100，FP32、关闭TF32；真实clean chunk5、layer1/15/29；3次预热、10次交错同步计时；输入预驻留。计入冻结Risk特征计算、预测特征构造和选定一次交换，排除数据搬运、缓存提交和attention。使用未融合Python参考实现，不作为生产高效kernel成本或整层吞吐结论。',
 'layer | Risk参考ms | 预测原型总ms | 增量ms | CUDA/CPU mask一致']
 for r in gpu['rows']:lines.append(f"{r['layer']} | {r['risk_reference_ms']:.3f} | {r['predictor_total_ms']:.3f} | {r['extra_ms']:.3f} | {r['cuda_cpu_mask_equal']}")
 lines += ['', '验收与局限',f"144例、{audit['total_masks_checked']}个方法/步骤mask的own、预算和FP64目标独立重算通过；734个原冻结文件SHA256不变。在线摘要质量与冻结参考最大绝对差{audit['max_summary_mass_error']:.2e}；教师字段投毒测试通过。所有运行包含未改善、误判正收益和退化样本，不事后剔除。",'T4已完成最小线性原型及离线验证；未训练大型网络、未做部署集成、未做下游视频/质量测试。GPU FP32 mask一致性只验证上述3个成本实例；全144例主结果为FP64离线目标。','',
 '四个研究问题的答案',
 'Q1：本两场景流式状态下，集合交换收益普遍存在；泛化范围仅限所采样场景/轨迹。',
 'Q2：前四轮取得大部分局部收益；全局最优恢复率仍需已有逐实例界，不能以本轮full-four-step oracle冒充OPT。',
 'Q3：Medium/Large简单Risk候选池能够保留多数有效收益；Small更便宜但损失明显。',
 'Q4：16维真实有符号表示保留较多决策信息，32维更接近完整搜索；低维可表示不等于可预测。本线性预测器失败，下一步若继续，应先研究残差方向和质量比的可预测性，而不是扩大在线交换轮数。','',
 '复现入口：analyses/t3_t4_v1/protocol.json、selected_model.json、audit.json、t3/t4_summary.csv、t3/t4_per_instance.csv、gpu_cost.json。逐实例全部mask/轨迹保存在t3、t3_paired、t4；原T1/T2入口及证书不变。模型、投影、代码及输入哈希随结果保留。']
 (ROOT/'REPORT_T1_T4.zh.txt').write_text('\n'.join(lines)+'\n')
 # Compact research plots: evaluation only, fixed 4-step oracle baseline and matched one-step T4.
 fig,ax=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
 names=['full','small','medium','large'];vals=[row('full_q64_'+n)['mean_gain_pct'] for n in names];ax[0].bar(names,vals);ax[0].set(title='T3: true residual, 4 swaps',ylabel='Mean NMSE reduction vs Risk (%)')
 for kind in ['random','pca']:ax[1].plot([4,8,16,32],[row(f'{kind}_r{r}_q64')['mean_gain_pct'] for r in [4,8,16,32]],'o-',label=kind)
 ax[1].axhline(full['mean_gain_pct'],ls='--',color='gray',label='128D full');ax[1].set(title='T3: signed projection, Q64',xlabel='Residual dimensions',ylabel='Mean NMSE reduction (%)');ax[1].legend()
 vals=[0,row('full_q64_small',1)['mean_gain_pct'],row('pca_r16_q64_small',1)['mean_gain_pct'],sel['mean_gain_pct']];ax[2].bar(['Risk','True128','True16','Pred16'],vals,color=['gray','steelblue','teal','tomato']);ax[2].axhline(0,color='black',lw=.5);ax[2].set(title='T4: same small pool, one swap',ylabel='Mean NMSE reduction (%)')
 fig.suptitle('horse_field: 72 head-tiles, two trajectories; offline evaluation, not unseen-scene proof');fig.savefig(OUT/'overview_T3_T4.png',dpi=170);fig.savefig(OUT/'overview_T3_T4.pdf');plt.close(fig)
 # Persist exact runnable sources, weights, and versions without copying multi-GB contributions.
 code=OUT/'code'/'local_experiments'/'set_oracle_v1';code.mkdir(parents=True,exist_ok=True);source=Path(__file__).parent
 for name in ['__init__.py','common.py','problem.py','stage34.py','report34.py','benchmark34.py','audit34.py','diagnose34.py','final34.py']:
  if (source/name).resolve()!=(code/name).resolve():shutil.copy2(source/name,code/name)
 if not (OUT/'code'/'baseline_review_ops.py').exists():shutil.copy2('/tmp/group_response_178f95a8/local_experiments/review_v1/ops.py',OUT/'code'/'baseline_review_ops.py')
 dump(OUT/'provenance.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,manifest_sha256=sha(ROOT/'manifest.json'),sources={p.name:sha(p) for p in code.glob('*.py')},models={p.name:sha(p) for p in OUT.glob('*.npz')},train_ids=audit['splits']['train'],validation_ids=audit['splits']['validation'],evaluation_ids=audit['splits']['evaluation']))
 (OUT/'REPRODUCE.txt').write_text('Set SET_ORACLE_ROOT to the complete frozen dataset directory. Put code/ on PYTHONPATH. From the source repository: python -m local_experiments.set_oracle_v1.stage34 t3 --workers 24; stage34 paired; stage34 t4; report34; audit34; diagnose34; benchmark34; final34. Existing per-instance results are reused. To repeat with changed algorithms use a NEW output version, never overwrite contributions. prepare/train are needed only to rebuild projections/model weights from the declared TRAIN split; supplied NPZ weights are exact artifacts. Full T1 dataset required; review archive omits contributions, candidate surfaces and full certificates. All objectives are offline pre-W_O, no downstream deployment claim.\n')
 pack=ROOT.parent/'set_oracle_v1_T1_T4_review.tar.gz'
 with tarfile.open(pack,'w:gz') as tar:
  for name in ['REPORT_T1_T4.zh.txt','REPORT.zh.txt','SCHEMA.txt','identity_and_splits.json','acceptance.json'] :tar.add(ROOT/name,arcname='set_oracle_v1_T1_T4_review/'+name)
  for name in ['summary.json','per_instance.csv','global_bounds.csv','recovery_intervals.csv','overview.png','preregistered_exchange_curves.png']:tar.add(ROOT/'analyses'/name,arcname='set_oracle_v1_T1_T4_review/T1_T2/'+name)
  for p in sorted(OUT.iterdir()):
   if p.is_file() and p.suffix in ('.csv','.json','.txt','.npz','.png') and not p.name.endswith('_actions.csv'):tar.add(p,arcname='set_oracle_v1_T1_T4_review/T3_T4/'+p.name)
  tar.add(OUT/'code',arcname='set_oracle_v1_T1_T4_review/T3_T4/code')
 print('report',ROOT/'REPORT_T1_T4.zh.txt','archive_bytes',pack.stat().st_size)
if __name__=='__main__':main()
