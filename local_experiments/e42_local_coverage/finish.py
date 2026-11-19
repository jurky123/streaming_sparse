"""Bundle the fixed experiment, including a stop decision, without new training."""
from .core import *
import shutil,tarfile

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def finish():
 audits=[json.loads((D/f'audit_s{s}.json').read_text()) for s in [17,29]];assert all(a['passed'] for a in audits);gates=json.loads((D/'final_gates.json').read_text());rows=read(D/'decision_summary.csv');info=read(D/'information_summary.csv');coverage=read(D/'coverage_summary.csv');curves=read(D/'training_gradient_summary.csv');head=[r for r in rows if r['dimension']=='overall'];txt=[]
 def put(s):txt.append(s)
 put('# E42：固定真实质量补充的局部响应监督\n')
 put('**一次固定权重对照已完成。预检通过，复用E39-T0；仅新增两个种子的T1，各2304次更新，主结果固定最终epoch1。没有新增推理信息、模型容量或缓存，Test封存。**\n')
 put('机制门槛：**%s**；部署门槛：**%s**。'%(('通过' if gates['mechanism'] else '未通过'),('通过' if gates['deployment'] else '未通过')))
 put('\n## 1 单变量和T0复用契约\n')
 put('每Query固定`w1=0.5*m²+0.5*sum(m²)*alpha_true²/sum(alpha_true²)`，权重停止梯度，逐Query总权重不变、每页保留至少半数原权重。仅修改局部响应loss；原局部尺度、β Huber、gain/listwise集合损失、Adam、clip=1、C/φ/B16、M16、Risk/Medium/own全部保持。真实质量只用于离线监督，不进入Reader或动作排序。沿用旧评价中的真实Dense范数作为离线分数单位：它是全候选共享的正数，不影响排序、正收益判断或在线决策；不把它视作可部署输入。\n')
 put('两个种子的T0初始化、C/φ与状态计划SHA核对通过。各丢弃式重放16步T0，local/quality/set loss、总loss、梯度范数和gain scale对历史记录的最大差均为0；没有重训完整T0。T0开发集动作和真实收益也逐例重放通过。新增正式训练4608步，另32步仅用于数值契约核验并丢弃。T1全部参数均由原E29初始化，Adam状态重置；保存epoch0但不评价选择它。\n')
 put('本轮关键结果：训练Set恢复T0→T1为5.36%→6.70%、11.39%→11.63%；开发为13.89%→13.79%、17.17%→17.89%。seed17开发收益略降，seed29训练H_loss略升，因此两种子均未同时满足机制条件。低估子集响应MSE相对T0在训练下降约1.20%/1.07%，开发下降约0.94%/0.78%；大幅增加监督份额没有转化为同等幅度的预测或决策改善。\n')
 put('## 2 冻结预检：监督确实补到了哪里\n')
 put('份额先按事件归一化、再对该子集非空事件等权平均；低估定义固定为m<alpha_true/10，不跨种子池化。\n')
 put('|seed|低估子集|原局部loss份额|新局部loss份额|原梯度L2份额|新梯度L2份额|\n|---|---|---:|---:|---:|---:|')
 for seed in ['17','29']:
  for role in ['all','harmful_predicted_endpoint','oracle_endpoint']:
   pair={r['arm']:r for r in coverage if r['seed']==seed and r['role']==role and r['status']=='under10'};a,b=pair['T0'],pair['T1'];put('|%s|%s|%.4f%%|%.4f%%|%.4f%%|%.4f%%|'%(seed,role,100*float(a['loss_share']),100*float(b['loss_share']),100*float(a['gradient_l2_share']),100*float(b['gradient_l2_share'])))
 put('\n权重守恒、半权重下界、解析输出梯度均通过。每种子8个预定head9事件加2个既有最差事件，共10个head9快照；新旧权重各有9个触发全局裁剪。隐藏层local范数中位数seed17为3.248→3.186，seed29为2.827→2.780。该样本不是长期训练过程，不能用来证明或排除长期梯度冲突。\n')
 put('## 3 完整训练与开发决策\n')
 put('每种子1152个训练事件（8轨迹）、288个开发事件（2轨迹）。E29决策复用E41，T0/T1均完整冻结评价。相关事件不作为独立视频。真实收益仍用Q64完整目标，评分M16。\n')
 put('|seed|划分|模型|策略|Oracle恢复|H_gain|H_loss|退化数|P10相对改善|最大1/5损害占比|\n|---|---|---|---|---:|---:|---:|---:|---:|---:|')
 for r in head:
  put('|%s|%s|%s|%s|%.2f%%|%.6f|%.6f|%s/%s|%.2f%%|%.1f%% / %.1f%%|'%(r['seed'],r['split'],r['arm'],r['policy'],float(r['recovery_pct']),float(r['H_gain']),float(r['H_loss']),r['harm'],r['events'],float(r['p10_relative_pct']),100*float(r['max1_harm_share']),100*float(r['max5_harm_share'])))
 put('\n下表直接对照同预算T0，避免把续训本身的变化归给新权重。\n')
 put('|seed|划分|T1−T0恢复百分点|T1−T0净收益|T1−T0 H_loss|\n|---|---|---:|---:|---:|')
 for seed in ['17','29']:
  for split in ['train','development']:
   a=next(r for r in head if r['seed']==seed and r['split']==split and r['arm']=='T0' and r['policy']=='Set');b=next(r for r in head if r['seed']==seed and r['split']==split and r['arm']=='T1' and r['policy']=='Set');put('|%s|%s|%+.2f|%+.6f|%+.6f|'%(seed,split,float(b['recovery_pct'])-float(a['recovery_pct']),float(b['gain'])-float(a['gain']),float(b['H_loss'])-float(a['H_loss'])))
 put('\n逐轨迹Set结果（训练8条、开发2条均保留）：\n')
 put('|seed|划分|轨迹|T0恢复|T1恢复|T1净收益|T1 H_loss|\n|---|---|---|---:|---:|---:|---:|')
 for r in rows:
  if r['dimension']=='trajectory' and r['arm']=='T1' and r['policy']=='Set':
   a=next(z for z in rows if z['seed']==r['seed'] and z['split']==r['split'] and z['value']==r['value'] and z['dimension']=='trajectory' and z['arm']=='T0' and z['policy']=='Set');put('|%s|%s|%s|%.2f%%|%.2f%%|%.6f|%.6f|'%(r['seed'],r['split'],r['value'],float(a['recovery_pct']),float(r['recovery_pct']),float(r['gain']),float(r['H_loss'])))
 put('\n逐例平均和最差绝对NMSE见`decision_summary.csv`，保留原初始误差>1e−12口径，无截尾或剔除异常。Top1/4/8为事后覆盖，不能当成线上收益。Set与Independent共享所有预测和候选信息，Set优势必须与no-op和匹配预算T0同时比较。\n')
 put('## 4 响应与β是否同步改善\n')
 put('固定E29旧端点和低估子集，避免按新模型错误重新选样。下表是事件等权的局部响应MSE与β MAE；这些是信息诊断，不替代实际收益。E29响应误差复用E40缓存切片，T0/T1均使用同一M16前向。\n')
 put('|seed|划分|旧子集|模型|局部MSE|β MAE|预测质量L1|\n|---|---|---|---|---:|---:|---:|')
 for r in info:
  if r['dimension']=='overall' and r['status']=='under10' and r['role'] in ['all','harmful_predicted_endpoint','oracle_endpoint']:
   put('|%s|%s|%s|%s|%.6g|%.6g|%.6g|'%(r['seed'],r['split'],r['role'],r['arm'],float(r['local_mse_mean']),float(r['beta_mae_mean']),float(r['mass_l1_mean'])))
 put('\nβ监督未变不等于β预测不变：共享隐藏层同时更新。应以表中实际变化为准，不将任何β偏移视作额外干预。所有layer/head分列均保存在`information_summary.csv`，不能仅由layer29的大误差主导结论。\n')
 put('## 5 训练梯度与优化尺度\n')
 put('|seed|模型|epoch|梯度范数中位数|P90|裁剪比例|\n|---|---|---:|---:|---:|---:|')
 for r in curves:
  if r['dimension']=='overall':put('|%s|%s|%s|%.4g|%.4g|%.2f%%|'%(r['seed'],r['arm'],r['epoch'],float(r['gradnorm_median']),float(r['gradnorm_p90']),100*float(r['clip_fraction'])))
 put('\n总权重不变不意味着loss值或梯度相同。新权重改变误差采样位置，继而改变全局裁剪和Adam更新；这是该固定干预的实际作用，不声称隔离了所有优化尺度影响。局部loss定义不同，其数值不直接作为训练优劣指标。完整曲线及分layer/head表保留，不进行后验调权。\n')
 put('## 6 预登记判定与停止\n')
 for r in gates['seeds']:put('- seed%d：机制条件 %s；部署条件 %s。'%(r['seed'],json.dumps(r['checks'],ensure_ascii=False),json.dumps(r['deployment_checks'],ensure_ascii=False)))
 if not gates['mechanism']:put('\n**未通过两种子共同机制门槛，停止这一固定权重配置。** 预检覆盖提高本身不构成成功；不能通过更换混合系数、追加epoch、修改β/KL/gate来挽救本轮。结果最多界定本次覆盖修正的效果，不证明所有覆盖方式或R4表示不可能有效。\n')
 elif not gates['deployment']:put('\n覆盖机制门槛通过，但可靠性/部署门槛未全部通过，记录为有限进展；Test仍封存，不自动添加新的修正模块或扩大训练搜索。\n')
 else:put('\n两层门槛均通过；本轮仍保持Test封存，已完成预定开发验证，不自动扩展新算法或测试范围。\n')
 put('## 7 审计、成本与阅读顺序\n')
 put('来源、T0完整开发重放、T1优化器步数/设置、初始化、冻结摘要/φ、count/own、正质量混合评分独立公式及最差动作直接目标均已检查。增量真实gain的减法抵消误差单独报告（rtol1e−5、atol1e−9），保留原训练标签；不隐藏极端事件。\n')
 for a in audits:put('- seed%d：独立检查%d事件，最大评分差%.3g，直接真实gain差%.3g。'%(a['seed'],a['events'],a['max_score_error'],a['max_gain_error']))
 put('\n线上参数51,157、原始KV读取、页面缓存、M16读出和Medium搜索均不变；新权重只在训练使用。没有重新执行原始Attention，没有Test读取。\n\n阅读顺序：本报告 → `decision_summary.csv` → `information_summary.csv` → `coverage_summary.csv`、head9表 → `training_gradient_summary.csv` → `final_gates.json`、`protocol.json`和审计。事件级明细和两个T1最终模型在审阅包中；复用的T0模型仍位于E39 artifact。\n')
 (D/'REPORT.md').write_text('\n'.join(txt));dump(D/'stage_status.json',dict(preflight='passed',T0='reused',T1_updates_per_seed=2304,final_epoch=1,mechanism=gates['mechanism'],deployment=gates['deployment'],test='sealed',followup_training=False,online_cost_change=0))
 evidence=ROOT/'docs/research/evidence/e42_local_coverage_v1';evidence.mkdir(parents=True,exist_ok=True);small=['REPORT.md','protocol.json','stage_status.json','preflight_gate.json','final_gates.json','coverage_summary.csv','decision_summary.csv','information_summary.csv','training_gradient_summary.csv']
 for s in [17,29]:small += [f'contract_s{s}.json',f'contract_prefix_s{s}.csv',f'head9_gradients_s{s}.csv',f'head9_checks_s{s}.csv',f'preflight_s{s}.json',f'audit_s{s}.json',f'audit_events_s{s}.csv']
 mp=ROOT/'docs/research/evidence/manifest.json';manifest=json.loads(mp.read_text());manifest=[r for r in manifest if 'e42_local_coverage_v1/' not in r['snapshot']]
 for name in small:
  shutil.copy2(D/name,evidence/name);manifest.append(dict(source=str((D/name).relative_to(ROOT)),snapshot=str((evidence/name).relative_to(ROOT)),sha256=sha(D/name)))
 dump(mp,manifest);package=[D/n for n in small]
 for seed in [17,29]:
  for arm in ['T0','T1']:
   for n in ['decisions.csv','information.csv','evaluation_complete.json']:package.append(D/f'{arm}_s{seed}'/n)
  for n in ['training_complete.json','curves.json','epoch1.pt']:package.append(D/f'T1_s{seed}'/n)
 dump(D/'package_manifest.json',[dict(file=str(p.relative_to(D)),sha256=sha(p)) for p in package])
 with tarfile.open(D/'e42_review.tar.gz','w:gz') as tf:
  for p in package+[D/'package_manifest.json']:tf.add(p,arcname=str(p.relative_to(D)))
 print('PACKAGED',len(package),flush=True)
if __name__=='__main__':finish()
