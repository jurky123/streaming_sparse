from .report import *
import shutil,tarfile

def main():
 audit=json.loads((DEST/'independent_audit.json').read_text());continuation=json.loads((DEST/'continuation_checks.json').read_text());rows=list(csv.DictReader((DEST/'decision_summary.csv').open()));count=sum(r['decisions'] for r in audit['counts'])
 provenance=dict(protocol_sha=sha(DEST/'protocol.json'),dataset_manifest_sha=sha(OUT/'dataset_manifest.json'),fit_sha=sha(OUT/'fit.pt'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()},dependency_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'local_experiments/e30_beta_diagnostics/run.py',ROOT/'local_experiments/e29_local_reader/core.py',ROOT/'local_experiments/e27_adaptive_query/core.py',ROOT/'local_experiments/e26_oracle_coefficients/core.py',ROOT/'local_experiments/review_v1/ops.py']},runs=[json.loads((DEST/f's{s}'/'complete.json').read_text()) for s in [17,29]],training=False,test_accessed=False)
 dump(DEST/'provenance.json',provenance)
 costs=dict(new_network_parameters=0,new_raw_KV_probes=0,new_cache_bytes=0,training_updates=0,deployed_code_changed=False,feature_diagnostic_only=True,prospective_variance_feature_GFLOPs_upper=(864*16*410*72)/1e9,note='Upper main-arithmetic estimate if q2 already computed but subgroup variance and weighted reduction not reused; existing Risk can reuse its variance contractions. Excludes scalar normalization, IO, launch cost; no latency or deployed speed claim.')
 dump(DEST/'cost_scope.json',costs)
 teacher=list(csv.DictReader((DEST/'teacher_gap.csv').open()));ts=[]
 for seed in [17,29]:
  for state in range(5):
   ss=[r for r in teacher if int(r['seed'])==seed and int(r['state'])==state];gain=sum(float(r['teacher_true_gain']) for r in ss);ora=sum(float(r['actual_oracle']) for r in ss);ts.append(dict(seed=seed,state=state,events=len(ss),teacher_real_gain=gain,actual_oracle_gain=ora,recovery=gain/ora if ora else None,regret=ora-gain,negative=sum(float(r['teacher_true_gain'])< -1e-12 for r in ss),noop=sum(r['teacher_execute']=='False' for r in ss)))
 table('teacher_gap_summary.csv',ts)
 report=DEST/'REPORT.md';text=report.read_text();opening='''## 本轮结论

**权重不均衡和β特征不匹配都提供了部分解释，但本轮没有得到可靠的跨Query β复用方案，不启动β训练。**

1. 原mass²拟合平均把约75.4%的权重集中在单个出现上下文，约34.3%的历史页有一个上下文占比超过90%。A1事件均衡在seed29大幅改善Shared，但在seed17没有同样改善；LOTO只升到10.99%/9.47%。所以权重主导是真实问题，却不是充分解释。
2. A2固定Key方差替换把Shared提升到31.55%/43.27%，LOTO提升到15.09%/24.93%，均优于相同均衡权重的旧特征。但LOTO依然远低于TrueBeta，seed17的Set低于同信息Independent、逐例平均仍负、条件汇总相关更低；seed29退化92例，未改善原91例。因此不能据此开始在线训练或部署。
3. 严格过去chunk、三类共同页面支持集上，A2恢复仅0.22%/−3.91%，对应同事件相同背景基线20.51%/12.19%。覆盖仅约15%的历史页出现、约10%的历史mass；说明这个固定特征在已覆盖的长期复用测试上未成功，不能外推所有历史页，更不能用不等覆盖的分类结果解释Query分布。
4. 事件内跨方法的真实regret相关中位数：质量回归损失0.179/0.293，决策蒸馏损失0.295/0.263。蒸馏在seed17更对齐，在seed29没有一致优势；这只是冻结方案间的相关分析，未证明优化它会提高实际质量。
5. 主LOTO与Shared仍使用教师系数，且当前页保留逐事件Oracle；没有测得R4 Page Encoder的在线新能力。主LOTO可以使用其他已保存的后续chunk，只有forward子诊断严格禁止未来donor。所有诊断都保持Test封存。

'''
 text=text.replace('## A：原权重、事件均衡与Key方差特征',opening+'## A：原权重、事件均衡与Key方差特征',1)
 text+='\n\n## 实现及数值审计\n\n'
 text+=f'全部{count}个Set/Independent动作通过count/own/候选池检查；E30基线、事件Oracle、True及A0 Shared/LOTO全部动作和收益复现。独立NumPy核验{len(audit["ridge"])}个共享/留出拟合，最大预测β差{max(r["error"] for r in audit["ridge"]):.3g}；对{len(audit["checks"])}个选中动作做longdouble正混合输出参考，预测评分最大绝对差{max(r["predicted_gain_error"] for r in audit["checks"]):.3g}、真实Q64收益最大绝对差{max(r["true_gain_error"] for r in audit["checks"]):.3g}。\n\n'
 text+='全部576例的tile/step/forward donor计数与共同支持集经过独立重建；所有forward donor严格来自更早chunk。Key方差用显式子组/坐标块循环重算，未读取真实logZ或输出构造该特征。C、phi、Reader哈希未变。决策蒸馏使用对称温度KL，TrueBeta自身损失独立检查为0；训练阶段不存在。\n\n'
 maxima=[max(r['feature_max'] for r in json.loads((DEST/f's{s}'/'audit.json').read_text())) for s in [17,29]]
 text+=f'A2保留原始量纲，观测最大Key方差特征约{max(maxima):.2f}，而旧tanh φ有界。这除了改变可用信号，也改变设计矩阵尺度及相对ridge的几何；本轮没有搜索归一化，因此不能把结果全部归因于信息量变化，也不能据此否定所有Key统计表示。\n\n'
 text+='两类质量数值和极端相对退化完整保存；集合评分沿用E30逐集合log-sum-exp路径，不裁剪β或删除难例。继承的R4/当前响应不变，不把因数学单位不同造成的损失下降当作训练成功。\n\n'
 text+='## TrueBeta教师仍有的决策缺口\n\n下表是五状态共同面板，不是初始Medium全部512动作，分母不能与主表直接互换。状态0/1/2/3/4为Risk、教师1/2步、两个合法扰动。\n\n|seed|状态|教师实际收益恢复|退化/no-op|\n|---|---|---:|---|\n'
 for r in ts:text+=f"|{r['seed']}|{r['state']}|{r['recovery']:.2%}|{r['negative']}/{r['noop']}|\n"
 text+='\n## 成本和交付范围\n\n没有部署模块、缓存或新增参数。若未来仅替换β特征，总五项容量不增加；已有Risk能复用q²和Key方差收缩。不能把离线原型计算时间当线上延迟，也没有实测加速。精简包只包含报告、主要表格、协议、审计与源码，完整逐事件β拟合和候选面板保留在s17/s29。\n'
 report.write_text(text)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4));methods=['Predicted','A0-LOTO','A1-LOTO','A2-LOTO','True']
 for ax,seed in zip(axes,[17,29]):
  yy=[100*float(next(r['R_executed'] for r in rows if int(r['seed'])==seed and r['method']==m and r['policy']=='Set')) for m in methods];ax.bar(range(5),yy,color=['#999999','#bd8842','#7591b5','#278b83','#c2a24c']);ax.set_xticks(range(5),['E29','Original\nLOTO','Balanced\nLOTO','Key-var\nLOTO','True beta']);ax.set_ylabel('Q64 oracle gain recovery (%)');ax.set_title(f'Frozen C seed {seed}');ax.axhline(0,color='black',lw=.7)
 fig.tight_layout();fig.savefig(DEST/'reuse_quality.png',dpi=180);plt.close(fig)
 include=['REPORT.md','protocol.json','provenance.json','cost_scope.json','continuation_checks.json','independent_audit.json','decision_summary.csv','decision_per_event.csv','context_decision_summary.csv','coverage_summary.csv','quality_summary.csv','loss_summary.csv','loss_regret_correlations.csv','within_event_summary.csv','conditional_summary.csv','weight_dominance_summary.csv','generalization.csv','teacher_gap_summary.csv','tail_cases.csv','reuse_quality.png']
 stage=DEST/'review';stage.mkdir(exist_ok=True)
 for name in include:shutil.copy2(DEST/name,stage/name)
 shutil.copytree(Path(__file__).parent,stage/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(stage/'SHA256SUMS.json',{str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'})
 with tarfile.open(DEST/'e31_review.tar','w') as tar:tar.add(stage,arcname='e31_review')
 print('FINISHED',DEST/'e31_review.tar',flush=True)
if __name__=='__main__':main()
