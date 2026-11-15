from .report import *
import shutil,tarfile

def finish():
 report();gate=json.loads((E27/'B_gate.json').read_text())
 if gate['passed']:
  from .report_b import report_b
  report_b()
  assert (E27/'B_independent_audit.json').exists()
 costs=list(csv.DictReader((E27/'cost_estimate.csv').open()));rows=list(csv.DictReader((E27/'decision_summary.csv').open()));fitrows=list(csv.DictReader((E27/'fit_per_event.csv').open()));fits=[];audits=[]
 for seed in [17,29]:
  dest=E27/f's{seed}';d=json.loads((dest/'training_complete.json').read_text());a=json.loads((dest/'analysis_complete.json').read_text());assert d['frozen_C_exact'];assert a['selected_sha']==sha(dest/'selected.pt');audits.append(dict(training=d,analysis=a))
  for method in ['Adaptive64','Adaptive16','Untrained64']:
   ff=[r for r in fitrows if int(r['seed'])==seed and r['method']==method];fits.append(dict(seed=seed,method=method,z_nmse64_median=float(np.median([float(r['z_nmse64']) for r in ff])),z_nmse16_median=float(np.median([float(r['z_nmse16']) for r in ff])),unselected_z_nmse_median=float(np.median([float(r['unselected_z_nmse64']) for r in ff])),phi_saturated_fraction_mean=float(np.mean([float(r['phi_saturated_fraction']) for r in ff])),design_rank_mean=float(np.mean([float(r['mean_design_rank']) for r in ff]))))
 table('fit_summary.csv',fits)
 # Check replay against selected development checkpoint for all initial actions.
 for seed in [17,29]:
  dest=E27/f's{seed}';done=json.loads((dest/'training_complete.json').read_text());ref={r['event']:r for r in json.loads((dest/f'development_epoch{done["best"]["epoch"]}.json').read_text())};dec=json.loads((dest/'decisions.json').read_text())
  for r in dec:
   if r['method']!='Adaptive64':continue
   old=ref[r['event']];key='set' if r['policy']=='Set' else 'independent';assert r['remove']==old[key+'_remove'] and r['add']==old[key+'_add'] and r['execute']==old[key+'_execute'];assert abs(r['executed_gain']-old[key+'_gain'])<1e-12
 B_runs=[json.loads((E27/f'B_s{s}'/'training_complete.json').read_text()) for s in [17,29]] if gate['passed'] else []
 dump(E27/'audit.json',dict(runs=audits,B_runs=B_runs,selected_replay_exact=True,stage0_passed=json.loads((E27/'stage0.json').read_text())['gradcheck'],B_executed=gate['passed'],reason='Conditional B completed' if gate['passed'] else 'both-seed reference gate not passed',test_accessed=False))
 with (E27/'REPORT.md').open('a') as f:
  f.write('\n## 本轮执行边界\n\n'+('两个种子通过A门槛，已执行预定的B系数预测实验；最终在线信息结果见B表。' if gate['passed'] else '两个种子未同时通过A门槛，因此未执行B。')+'A表Adaptive始终是“学习φ＋真实系数＋真实beta”的离线能力实验，不可与B的预测系数结果混称在线收益。未增加维度、网络规模、训练轮数或新数据，Test保持封存。\n')
  if gate['passed']:
   br=list(csv.DictReader((E27/'B_decision_summary.csv').open()))
   f.write('\n## 综合判断\n\nA的学习特征在Oracle系数/真实beta下通过两个种子的门槛；B未稳定恢复这部分能力。\n')
   for seed in [17,29]:
    pred=next(r for r in br if int(r['seed'])==seed and r['method']=='PredictedBeta' and r['policy']=='Set');true=next(r for r in br if int(r['seed'])==seed and r['method']=='TrueBeta' and r['policy']=='Set')
    f.write(f"\n- seed{seed}：B实际Oracle恢复{float(pred['R_executed']):.2%}，逐例平均改善{float(pred['mean_improvement_pct']):+.2f}%，退化{pred['degraded']}/288；真实beta替换恢复{float(true['R_executed']):.2%}。\n")
   f.write('\nSet虽在汇总上优于同信息Independent和no-op，但大量退化及负的逐例均值仍不支持可靠在线决策。完美beta替换不能修复预测z的错误，也可能破坏两类预测误差的补偿；不能因此声称beta误差已经不重要。当前剩余瓶颈仍在有符号系数/质量的一致预测与动作排序。B采用新的损失归一化，因此与E25-D的差异不能全部归因于phi变化。按计划停止扩大该分支，不增加网络、rank、训练轮数、数据或Kernel实验。\n')
  f.write('\n重建损失下降不保证真实交换收益提高；checkpoints.csv保留随机初始化及全部四个epoch，按约定的开发汇总收益选择训练检查点。两个实验种子同时对应不同冻结C摘要，因此差异不能全部归因于网络初始化。未通过本次结构和训练方案也不证明R4完全没有可用信息。\n')
  f.write('\n数值修订：A训练后的审计发现大目标相减损失精度，因此改用2E·deltaO−||deltaO||²的等价增量式；初始和全部四个checkpoint重新评价并选择，训练权重未重跑。E26固定φ/SVD对照也在本轮用同一公式重算；修订前记录在before_stable_gain目录。因此当前恢复率可能与早期进度消息或旧E26表略有不同。\n')
  ca=json.loads((E27/'condition_precision_audit.json').read_text());f.write(f"\n补充反事实精度检查：90项极端状态动作中，最大相对差为{ca['max_relative_error']:.3g}；{ca['strict_exceptions']}项略超过1e-8+1e-7|参考值|的严格阈值，收益符号分歧{ca['sign_disagreements']}项。这些记录全部保留，不将此补充检查写成无例外通过。初始动作的独立审计与此分开。\n")
  f.write('\n所有初始动作已与训练阶段保存的检查点评价逐例复核；独立增强最小二乘和long-double动作检查见independent_audit.json。条件面板/完整模型保存在s17/、s29/；精简包不复制旧QKV或全部可重建面板。\n')
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  names=['Cphi_trueBeta','Untrained64','Adaptive64','SVD5_trueBeta','P1'];labels=['Fixed Cφ','Untrained adaptive','Learned adaptive','SVD rank5','True 16D']
  fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(names));
  for j,seed in enumerate([17,29]):
   vals=[100*float(next(r for r in rows if int(r['seed'])==seed and r['method']==name and r['policy']=='Set')['R_executed']) for name in names];bars=ax.bar(x+(j-.5)*.34,vals,.34,label=f'Seed {seed}');ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
  ax.set_xticks(x,labels);ax.set_ylabel('True Q64 oracle gain recovered (%)');ax.set_title('E27-A: Oracle coefficients + true quality correction, M16 decisions');ax.legend();ax.set_ylim(0,80);fig.tight_layout();fig.savefig(E27/'capability.png',dpi=180);plt.close(fig)
 except ImportError:pass
 code=ROOT/'local_experiments/e27_adaptive_query';dump(E27/'provenance.json',dict(source={p.name:sha(p) for p in code.glob('*.py')},adaptive_selected={str(s):sha(E27/f's{s}'/'selected.pt') for s in [17,29]},C={str(s):sha(checkpoint('C',s)) for s in [17,29]},fit_sha=sha(OUT/'fit.pt'),protocol_sha=sha(E27/'protocol.json'),E24_manifest_sha=sha(OUT/'dataset_manifest.json'),test_accessed=False,B_executed=gate['passed']))
 pkg=E27/'review';pkg.mkdir(exist_ok=True)
 for p in E27.iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md','.png']:shutil.copy2(p,pkg/p.name)
 for seed in [17,29]:
  for name in ['initial','selected']:shutil.copy2(E27/f's{seed}'/f'{name}.pt',pkg/f's{seed}_{name}.pt')
 if gate['passed']:
  for seed in [17,29]:shutil.copy2(E27/f'B_s{seed}'/'selected.pt',pkg/f'B_s{seed}_selected.pt')
 for folder in ['e27_adaptive_query','e26_oracle_coefficients','e25_decision_diagnostics','e24_joint_representatives']:
  shutil.copytree(ROOT/'local_experiments'/folder,pkg/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(E27/'e27_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e27_review')
 print('PACKAGED',E27/'e27_review.tar.gz')
if __name__=='__main__':finish()
