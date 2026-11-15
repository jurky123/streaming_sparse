from .report import *
import tarfile,shutil

def main():
 audit=json.loads((DEST/'independent_audit.json').read_text());gate=json.loads((DEST/'capacity_gate.json').read_text());summary=list(csv.DictReader((DEST/'decision_summary.csv').open()));signed=[]
 for seed in [17,29]:signed+=json.loads((DEST/f's{seed}'/'signed_components.json').read_text())
 table('signed_components_per_event.csv',signed);ss=[]
 for seed in [17,29]:
  for method in METHODS:
   sub=[r for r in signed if r['seed']==seed and r['method']==method];ss.append(dict(seed=seed,method=method,I_sign_accuracy_mean=float(np.mean([r['I_sign_accuracy'] for r in sub])),B_log_error_median=float(np.median([r['B_log_abs_error_mean'] for r in sub if r['B_log_abs_error_mean'] is not None])),I_unrepresentable_fp64_actions=sum(r['I_unrepresentable_fp64'] for r in sub),B_unrepresentable_fp64_actions=sum(r['B_unrepresentable_fp64'] for r in sub)))
 table('signed_components_summary.csv',ss)
 count=sum(r['decisions'] for r in audit['counts']);replay=list(csv.DictReader((DEST/'baseline_replay.csv').open()));mismatch=[r for r in replay if r['actions_match']!='True'];table('numerical_action_differences.csv',mismatch)
 cost=dict(stage='analytic diagnostics only',training_updates=0,new_online_module=False,raw_KV_probing_added=False,conditional_beta_training_executed=False,deployed_cost_change=0,note='No deployed changes; A reads frozen teacher data offline. No latency claim.')
 dump(DEST/'cost_scope.json',cost)
 provenance=dict(protocol_sha=sha(DEST/'protocol.json'),numerical_addendum_sha=sha(DEST/'numerical_addendum.json'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()},runs=[json.loads((DEST/f's{s}'/'complete.json').read_text()) for s in [17,29]],dataset_manifest_sha=sha(OUT/'dataset_manifest.json'),projection_fit_sha=sha(OUT/'fit.pt'),test_accessed=False,training_executed=False)
 dump(DEST/'provenance.json',provenance)
 report=DEST/'REPORT.md'
 with report.open('a') as f:
  f.write('\n\n## 数值修复与独立审计\n\n首次运行在极端拟合β下，exp(logmass+β)重构z溢出，旧公共缩放导致混合分母下溢。正式结果改为分别归一化预测全页和每个混合集合：预测精确页输出为摘要页输出＋冻结局部响应，质量用log-sum-exp；与解析局部模型在精确算术下等价。没有裁剪β、调整ridge或删除实例。Independent能量在必要时省略所有动作共享的正比例系数，保持排序和no-op。\n\n')
  f.write(f'预算/own/面板检查：{count}个决策全部通过。独立longdouble混合输出复核{len(audit["checks"])}个选中动作，真实收益最大绝对差{max(r["true_gain_error"] for r in audit["checks"]):.3g}；预测评分最大绝对差{max(r["predicted_score_error"] for r in audit["checks"] if r["predicted_score_error"] is not None):.3g}。NumPy独立ridge复核{len(audit["ls"])}页。冻结模型哈希未变。\n\n')
  f.write(f'与E29原混合精度路径比较，{len(replay)}个基线/TrueBeta决策中有{len(mismatch)}个动作或no-op不同；差异表记录near-tie或数值路径变化，不隐瞒。容量门槛采用本轮统一稳定计算的Predicted与True，避免混用计算路径。\n\n')
  f.write('I/B在某些拟合β下可超出FP64表示范围，即使完整交换收益有限；原线性值指标标明数值无效，补充signed_components_summary.csv使用带符号log-sum-exp恢复I符号和B对数误差，保留所有动作。B_log_error是自然对数尺度误差，不是原B NMSE。\n\n|seed|β|I符号准确率|B对数误差中位|I超FP64动作|B超FP64动作|\n|---|---|---:|---:|---:|---:|\n')
  for r in ss:f.write(f"|{r['seed']}|{r['method']}|{r['I_sign_accuracy_mean']:.2%}|{r['B_log_error_median']:.3g}|{r['I_unrepresentable_fp64_actions']}|{r['B_unrepresentable_fp64_actions']}|\n")
  f.write('\n质量相对误差也以对数形式计算；若比值超出FP64，线性比值记为空并保存log比值，不以分母下限伪装有限精度。真实β零误差的log为负无穷，使用exact_zero_error标记。完整结果不排除这些事件。\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,seed in zip(axes,[17,29]):
  ms=['Predicted','OracleCoeff64','Shared','LOTO','True','Candidate-True','Background-True'];ys=[100*float(next(r['R_executed'] for r in summary if int(r['seed'])==seed and r['method']==m and r['policy']=='Set')) for m in ms]
  ax.bar(range(len(ms)),ys,color=['#268b86']+['#c49a40']*6);ax.axhline(100*next(g['target'] for g in gate['seeds'] if g['seed']==seed),ls='--',color='#777',label='70% gap target');ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(len(ms)),['E29','Oracle\ncoeff','Shared','LOTO','True','Candidate\ntrue','Background\ntrue'],rotation=35);ax.set_title(f'C seed {seed}');ax.set_ylabel('True Q64 oracle gain recovery (%)');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(DEST/'beta_capacity.png',dpi=180);plt.close(fig)
 include=['REPORT.md','protocol.json','numerical_addendum.json','capacity_gate.json','audit.json','independent_audit.json','provenance.json','cost_scope.json','decision_summary.csv','decision_per_event.csv','information_summary.csv','conditional_summary.csv','generalization.csv','sensitivity_summary.csv','sensitivity_per_event.csv','sensitivity_pages.csv','tail_cases.csv','signed_components_summary.csv','beta_capacity.png']
 stage=DEST/'review';stage.mkdir(exist_ok=True)
 for name in include:shutil.copy2(DEST/name,stage/name)
 if (DEST/'numerical_action_differences.csv').exists():shutil.copy2(DEST/'numerical_action_differences.csv',stage/'numerical_action_differences.csv')
 shutil.copytree(Path(__file__).parent,stage/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(stage/'SHA256SUMS.json',{str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'})
 with tarfile.open(DEST/'e30_a_review.tar','w') as tar:tar.add(stage,arcname='e30_a_review')
 print('FINISHED',gate,flush=True)
if __name__=='__main__':main()
