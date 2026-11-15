from .report import *

def finish():
 report()
 summary=list(csv.DictReader((E26/'decision_summary.csv').open()));fits=list(csv.DictReader((E26/'fit_summary.csv').open()));outliers=[];ranks=[]
 for seed in [17,29]:
  dest=E26/f's{seed}';fit=json.loads((dest/'fit.json').read_text())
  for name in ['OracleCoeff_Cphi','OracleCoeff_Dphi']:
   sub=[r for r in fit if r['method']==name];ranks.append(dict(seed=seed,method=name,mean_residual_design_rank=float(np.mean([r['z_unregularized_rank_mean'] for r in sub])),mean_beta_design_rank=float(np.mean([r['beta_unregularized_rank_mean'] for r in sub]))))
  for idx in range(288):
   p=np.load(dest/f'panel_{idx:04d}.npz');t=p['true_gain'];tc=t-t.mean(0);true_max=float(np.max(abs(tc)))
   for name in ['OracleCoeff_Cphi','OracleCoeff_Dphi','Cphi_trueBeta','SVD5_trueBeta']:
    v=p[name];v=v-v.mean(0);outliers.append(dict(seed=seed,method=name,event=str(p['event']),predicted_centered_max=float(np.max(abs(v))),true_centered_max=true_max,conditional_corr=stablecorr(tc,v)))
 table('design_rank.csv',ranks);table('conditional_amplitude.csv',sorted(outliers,key=lambda r:r['predicted_centered_max'],reverse=True))
 ep=json.loads((E26/'extreme_precision_audit.json').read_text());assert all(r['same_noop'] and (r['same_argmax'] or r['equivalent_tie']) for r in ep['rows'])
 with (E26/'REPORT.md').open('a') as f:
  f.write('\n## 本轮结论\n\n- 主OracleCoeff恢复39.30%/31.98%，只保留对应P1收益的70.44%/49.91%，两个种子均未通过80%参考门槛。D自身φ匹配拟合也只有34.84%/38.21%，不能用更换φ的比较掩盖缺口。\n- 真正的最佳rank5响应配合真实beta恢复52.23%/59.74%，约为P1的93.60%/93.24%。相同真实beta下，固定Cφ只能恢复39.17%/49.16%；这支持优先检查Query特征方向，而不是直接增加残差输出维度。\n- 质量拟合也影响决策：seed29换用真实beta从31.98%升至49.16%，seed17则基本不变。不能把主实验全部损失归因于残差因子化。\n- Oracle系数明显优于实际Page Encoder，但远未达到P1。这说明现有系数预测与固定φ重建都留有缺口；它不是任意五项系数的最优决策证明。\n- 固定φ残差重建NMSE的事件中位数约0.517，最佳rank5约0.0155；固定φ设计有效rank均值约4.52，单纯“只有五维”不足以解释与最佳rank5的差距。\n- 同信息下Set的汇总真实改善为5.45%/4.09%，Independent为4.43%/2.68%，说明教师系数提供的信息仍有可利用的组合收益；这是离线教师结果，不是线上方法。\n- Oracle系数的逐事件双重中心化相关中位数约0.639，优于D的0.265/0.251；但汇总交互相关接近零，正负翻转召回仅约11%–12%。必须同时保留两种结论：部分事件方向更有信息，跨状态收益幅度仍可能极端失配。\n')
  f.write('\n极端评分诊断：conditional_amplitude.csv完整列出所有事件，不排除大误差样本。对六个大幅度事件、共30个状态使用long-double重新评价全部合法候选，argmax（或等价数值平局）与no-op一致。因此不能把这些异常简单丢弃为FP64溢出。独立log-sum-exp检查的最大绝对分数差可较大，因为分数本身可达1e21；最大相对误差约2.85e-15，不能把该绝对差当作attention NMSE误差。\n')
  f.write('\n没有执行E26-B、没有更新四代表或Query投影、没有新增QKV、没有访问Test。没有进行正则或rank网格搜索；本轮只报告固定解析方案。\n\n复现入口：local_experiments/e26_oracle_coefficients/README.md。完整系数与状态面板保存在s17/和s29/；精简包只含代码、报告、逐例表与审计，不复制既有模型、原始QKV或全部系数。\n')
 code=ROOT/'local_experiments/e26_oracle_coefficients';artifact_hashes={str(p.relative_to(E26)):sha(p) for seed in [17,29] for p in (E26/f's{seed}').glob('*.npz')};dump(E26/'artifact_manifest.json',artifact_hashes)
 dump(E26/'provenance.json',dict(source={p.name:sha(p) for p in code.glob('*.py')},checkpoints={f'{g}_{s}':sha(checkpoint(g,s)) for g in ['C','D'] for s in [17,29]},E24_fit=sha(OUT/'fit.pt'),E24_manifest=sha(OUT/'dataset_manifest.json'),E26_protocol=sha(E26/'protocol.json'),ridge=RIDGE,training=False,test_accessed=False))
 pkg=E26/'review';pkg.mkdir(exist_ok=True)
 for p in E26.iterdir():
  if p.is_file() and p.suffix in ['.json','.csv','.md']:shutil.copy2(p,pkg/p.name)
 shutil.copytree(code,pkg/'source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 # Include the precise imported diagnostic and model helpers, not raw data.
 for folder in ['e24_joint_representatives','e25_decision_diagnostics']:
  shutil.copytree(ROOT/'local_experiments'/folder,pkg/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 with tarfile.open(E26/'e26_review.tar.gz','w:gz') as tf:tf.add(pkg,arcname='e26_review')
 print('FINISHED',E26/'e26_review.tar.gz')
if __name__=='__main__':finish()
