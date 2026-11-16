from .run import *
from ..e41_training_signal.report import decision_summary
import shutil

def report():
 rows=[];lines=['# E43：小型训练集页面系数与编码器可拟合性对照','', '固定48事件、96状态，各组2304次更新；仅训练轨迹内诊断，非泛化或部署评价。P为原Reader，L为离线共享页面系数表，Teacher为真实M16/B16参考。Test封存。','', '|seed|模型|阶段|Oracle恢复|H_gain|H_loss|局部MSE|β MAE|','|---|---|---|---:|---:|---:|---:|---:|']
 for seed in [17,29]:
  dest=D/f's{seed}';assert (dest/'complete.json').exists()
  for epoch in [-1,1]:
   with (dest/f'evaluation_{epoch}.csv').open() as f:rr=list(csv.DictReader(f))
   if epoch==-1:
    p=[r for r in rr if r['arm']=='P'];l=[r for r in rr if r['arm']=='L'];assert len(p)==len(l)
    for a,b in zip(p,l):assert abs(float(a['executed_gain'])-float(b['executed_gain']))<1e-8
   for arm in ['P','L','Teacher']:
    for policy in ['Set','Independent']:
     ss=[r for r in rr if r['arm']==arm and r['policy']==policy];z=decision_summary(ss);z.update(seed=seed,arm=arm,policy=policy,epoch=epoch,local_mse=float(np.mean([float(r['local_mse']) for r in ss])),beta_mae=float(np.mean([float(r['beta_mae']) for r in ss])));rows.append(z)
     if policy=='Set':lines.append('|%s|%s|%s|%.2f%%|%.6f|%.6f|%.5g|%.5g|'%(seed,arm,epoch,z['recovery_pct'],z['H_gain'],z['H_loss'],z['local_mse'],z['beta_mae']))
 lines+=['','全部结果仅描述这套固定优化过程的有限拟合能力。L拥有更多自由参数，不能称为可部署模型，也不能据失败证明表达不可能。固定状态顺序为锁定事件顺序、Risk后背景交换，96状态循环24次。背景交换端点在原Medium外，候选合法性与own均断言核验。页面身份使用轨迹/layer/head与完整摘要哈希；同一摘要输入强制共享，未额外依赖原始KV。此身份规则比物理KV身份更强地合并摘要相同页面，是本轮解释限制。','', '结果表为summary.csv；事件级两状态得分与信息误差见各种子evaluation表。原始模型文件留在artifacts，不加入Git；报告和小型CSV进入仓库。']
 table(D/'summary.csv',rows);(D/'REPORT.md').write_text('\n'.join(lines)+'\n');out=ROOT/'docs/research/evidence/e43_fit_v1';out.mkdir(parents=True,exist_ok=True)
 for n in ['REPORT.md','summary.csv']:shutil.copy2(D/n,out/n)
 for seed in [17,29]:
  for n in ['protocol.json','complete.json','evaluation_-1.csv','evaluation_1.csv']:shutil.copy2(D/f's{seed}'/n,out/f's{seed}_{n}')
 print('REPORT DONE')
if __name__=='__main__':report()
