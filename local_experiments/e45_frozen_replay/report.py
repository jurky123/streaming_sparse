from .run import *
import shutil,hashlib,tarfile,subprocess

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def report():
 rows=[];conds=[]
 for seed in [17,29]:
  dest=D/f's{seed}';assert json.loads((dest/'audit.json').read_text())['passed'];rs=read(dest/'decisions.csv')
  for arm in ['P','L','P_to_L']:
   for role in ['Risk','BgSwap_train','BgSwap_holdout','trained_combined']:
    for policy in ['Set','Independent','Reuse_other']:
     rr=[r for r in rs if r['arm']==arm and (r['state_role']==role or role=='trained_combined' and r['state_role']!='BgSwap_holdout') and r['policy']==policy];s=decision_summary(rr);rows.append(dict(seed=seed,arm=arm,state_role=role,policy=policy,**s))
  scores=np.load(dest/'scores.npz')
  for arm in ['P','L']:
   t=np.stack([np.stack([scores[f'{e}_truth_{i}'] for i in [0,1]]) for e in range(48)]);p=np.stack([np.stack([scores[f'{e}_{arm}_{i}'] for i in [0,1]]) for e in range(48)]);tc=t-t.mean(1,keepdims=True);pc=p-p.mean(1,keepdims=True);tc-=tc.mean(2,keepdims=True);pc-=pc.mean(2,keepdims=True);cr=read(dest/'conditional.csv');cr=[r for r in cr if r['arm']==arm];rec=[float(r['sign_flip_tp'])/float(r['sign_flip_true']) for r in cr if float(r['sign_flip_true'])>0];conds.append(dict(seed=seed,arm=arm,pooled_double_center_corr=float(np.corrcoef(tc.ravel(),pc.ravel())[0,1]),flip_recall_event_mean=float(np.mean(rec))))
 table(D/'summary.csv',rows);table(D/'conditional_summary.csv',conds)
 lines=['# E45：E43冻结补证与邻近状态留出','', '复用E43最终P/L，无训练、无新Query、无开发或Test评价。已见48事件/96状态；每事件预先固定一个新背景交换，共48个留出状态。留出只改变集合，不检验Query或轨迹泛化。','', '背景端点固定为原Medium之外第二个合法精算页与第二个合法近似页，保护own；未按真实收益挑选。协议在回放前落盘。','', '## P→L移植与身份','', '将最终P的FP32页面系数填入共享L表，后续保留相同BF16转换、φ和评分。两个种子各144状态的响应、β、全部512评分逐位一致；原E43最终P/L的96状态评分与E44补存矩阵也逐位一致。L能表达P在这批输入上的配置；原L较差只能说明有限优化未到达该配置，不能证明L表达空间更小。','', 'E44物理哈希审计重新核验原token/K/V和摘要，未发现摘要等价而KV不同的身份碰撞。本轮再次核验并复用；两个tile/集合状态共享同一组系数。','', '## 实际收益与状态依赖','', '|seed|模型|状态|策略|恢复率|H_loss|退化|P10|','|---|---|---|---|---:|---:|---:|---:|']
 for r in rows:
  if r['arm']=='P':lines.append('|%d|%s|%s|%s|%.2f%%|%.6f|%d/%d|%.2f%%|'%(r['seed'],r['arm'],r['state_role'],r['policy'],r['recovery_pct'],r['H_loss'],r['harm'],r['events'],r['p10_relative_pct']))
 lines+=['','Reuse_other在Risk上沿用BgSwap_train的完整评分和no-op判断，在BgSwap_train上沿用Risk偏好，在留出状态上沿用Risk偏好。候选动作页对相同，实际收益使用目标状态的真实Q64完整输出；因此能直接比较重新评分是否有额外价值，不向决策提供教师。','', '|seed|模型|汇总双中心相关|翻转召回事件均值|','|---|---|---:|---:|']
 for r in conds:lines.append('|%s|%s|%.4f|%.2f%%|'%(r['seed'],r['arm'],r['pooled_double_center_corr'],100*r['flip_recall_event_mean']))
 lines+=['','完整L及P_to_L结果在summary.csv，P_to_L应与P完全相同。逐状态决策保存action、proposal_index、移出/加入页、execute、初始NMSE、真实gain；完整评分、独立评分、真实动作gain、mask、r/a池保存于scores.npz。state_role直接分列Risk/训练BgSwap/留出BgSwap，不依赖奇偶解释。','', '## 对原评审的记录','', '保留E43的正面信号：小Reader可以在受限训练范围学到有用页面选择信息。Seed17原Set相对Independent优势基本未扩大，seed29最终Independent略高于Set，不能将恢复提高等同于组合交互能力提高。P局部未加权MSE上升不等于加权训练local loss上升；原E43没有分列三项曲线，该缺项不由后续实验追认补齐。','', 'Teacher是真实M16/B16评分规则而非Q64完整Oracle；P可在训练状态上超过它，不代表预测比真值准确。巨大逐例相对退化须与绝对NMSE并列，原极小初始误差事件全部保留。','', '## 复现归档与限制','', '补齐E43导入闭包中的未归档代码，dependency_manifest.json逐文件记录SHA。已有工作区中部分依赖相对Git存在改动，本轮不覆盖生产文件；精确实际源码闭包另存source_snapshot.tar.gz。只克隆代码仍缺原始模型和规范数据：本轮不声称裸仓库可完整重训。','', '本轮冻结回放不能证明未见Query、未见轨迹或线上可靠性，也不启动新的训练。邻近集合的收益仅是一次锁定的局部外推检查，不能根据结果增选更多状态。']
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n')
 manifest=json.loads((D/'dependency_manifest.json').read_text())
 for f in manifest['files']:
  p=ROOT/f
  manifest.setdefault('sha256',{})[f]=hashlib.sha256(p.read_bytes()).hexdigest()
 dump(D/'dependency_manifest.json',manifest)
 with tarfile.open(D/'source_snapshot.tar.gz','w:gz') as tf:
  for f in manifest['files']:tf.add(ROOT/f,arcname=f)
 dest=ROOT/'docs/research/evidence/e45_frozen_replay_v1';dest.mkdir(parents=True,exist_ok=True)
 for n in ['REPORT.md','summary.csv','conditional_summary.csv','dependency_manifest.json','source_snapshot.tar.gz']:shutil.copy2(D/n,dest/n)
 for seed in [17,29]:
  for n in ['protocol.json','audit.json','decisions.csv','transplant_checks.csv','conditional.csv']:shutil.copy2(D/f's{seed}'/n,dest/f's{seed}_{n}')
 with tarfile.open(D/'e45_review.tar.gz','w:gz') as tf:
  for p in D.rglob('*'):
   if p.is_file() and p.name!='e45_review.tar.gz':tf.add(p,arcname=str(p.relative_to(D)))
 print('REPORT DONE',flush=True)
if __name__=='__main__':report()
