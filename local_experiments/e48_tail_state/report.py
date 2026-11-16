from .run import *
from ..e41_training_signal.report import decision_summary
import shutil,tarfile

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def report():
 out=ROOT/'docs/research/evidence/e48_tail_state_v1';out.mkdir(parents=True,exist_ok=True);lines=['# E48：尾部解剖与代表状态/完整训练对照','', '冻结摘要Reader微调分支在本轮固定对照之后归档；不自动追加损失、网络、clip或学习率搜索。Test封存。尾部因果机制未预先认定，极值集中和β低估不等于证明两行pooling代码造成全部损害。','', '## 尾部事件','', '|seed|事件|角色|最大β|token top1质量中位数|真实页质量最大值|','|---|---|---|---:|---:|---:|']
 summaries=[]
 for seed in [17,29]:
  dest=D/f's{seed}';assert (dest/'trained_evaluation_complete.json').exists();r=read(dest/'tail.csv')
  for event in sorted({x['event'] for x in r}):
   for role in ['remove','add']:
    v=[x for x in r if x['event']==event and x['role']==role];lines.append('|%d|%s|%s|%.4f|%.4f|%.4f|'%(seed,event,role,max(float(x['beta']) for x in v),float(np.median([float(x['top1_token_mass']) for x in v])),max(float(x['true_mass']) for x in v)))
  cs=read(dest/'conditional.csv');a=sum(float(x['adaptive']) for x in cs);c=sum(float(x['conditional']) for x in cs);lines+=['',f'seed{seed}：更广四状态的真实条件价值占比 {100*c/a:.4f}%；48事件全部保留。共同候选池按索引固定，可能不同于原Medium；仅作为新协议诊断，不与旧Oracle分母混用。']
  rr=read(dest/'trained_decisions.csv')
  for split in ['train','development']:
   for policy in ['Set','Independent']:
    v=[x for x in rr if x['split']==split and x['policy']==policy];summaries.append(dict(seed=seed,split=split,policy=policy,**decision_summary(v)))
 lines+=['','## 完整分布纯setloss固定对照','', '从E29 selected出发，重置Adam lr1e−4，clip1，原E39状态计划2304次更新，仅setloss；没有新网络或特征。与原E39共享T0同预算，不能与从零训练4608步直接混称公平对照。','', '|seed|划分|策略|恢复|H_loss|退化|','|---|---|---|---:|---:|---:|']
 for r in summaries:lines.append('|%d|%s|%s|%.2f%%|%.6f|%d/%d|'%(r['seed'],r['split'],r['policy'],r['recovery_pct'],r['H_loss'],r['harm'],r['events']))
 lines+=['','## 解释边界','', '尾部直接计算原始64 token的真实logits、top1/top2条件质量、代表最大logit、页β及真实页mass，包含错误移出和加入页。初始Risk误差很小不支持直接断言初选漏掉关键页；可能是交换破坏了已精算页保护，须以角色表判断。','', '2*tanh确实限制pooling权重比不超过e4，16token中单个权重上界约0.785；但之后代表重中心化不是凸组合，不能据上界直接推导最终Key范数或β因果效应。本轮另做单页Key的tanh/重中心化2×2反事实，冻结Value/stats/旧Risk；baseline可逐位重建。结果并不支持简单去除两行就解决问题，也没有重训新摘要。','', '新四状态由无教师固定策略生成，所有事件保留，先测条件价值后冻结Pset回放。状态范围和共同动作池同时改变，不能把结果当真实部署状态分布的最终认证。原始矩阵见审阅包。完整分布训练结果也不是最终视频质量或速度，未解封Test。','', '未执行新的摘要/初选训练；这是后续独立分支，不由尾部相关性自动授权部署。']
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n');table(D/'summary.csv',summaries)
 for n in ['REPORT.md','summary.csv']:shutil.copy2(D/n,out/n)
 for seed in [17,29]:
  for n in ['protocol.json','complete.json','tail.csv','pooling_counterfactual.csv','conditional.csv','states.csv','trained_evaluation_complete.json','trained_checks.csv']:shutil.copy2(D/f's{seed}'/n,out/f's{seed}_{n}')
 with tarfile.open(out/'review.tar.gz','w:gz') as tf:
  for p in D.rglob('*'):
   if p.is_file() and p.suffix in ['.py','.csv','.json','.npz','.pt','.md']:tf.add(p,arcname=str(p.relative_to(D)))
 print('REPORT DONE',flush=True)
if __name__=='__main__':report()
