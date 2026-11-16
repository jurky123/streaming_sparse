from .core import *
from ..e41_training_signal.report import decision_summary
import shutil,tarfile

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run():
 summaries=[];conditional=[];physics=[];teacher=[]
 for seed in [17,29]:
  dest=D/f's{seed}';assert (dest/'complete.json').exists() and (dest/'physics_complete.json').exists()
  for step in [-1,0,384,768,1152,1536,1920,2304]:
   rs=read(dest/f'evaluation_{step}.csv');npz=np.load(dest/f'scores_{step}.npz');t=npz['true_gain'];assert t.shape==(96,512)
   for name in [k for k in npz.files if k!='true_gain']:
    p=npz[name];assert p.shape==(96,512) and np.isfinite(p).all();tc=t.reshape(48,2,512)-t.reshape(48,2,512).mean(1,keepdims=True);pc=p.reshape(48,2,512)-p.reshape(48,2,512).mean(1,keepdims=True);tc-=tc.mean(2,keepdims=True);pc-=pc.mean(2,keepdims=True);corr=float(np.corrcoef(tc.ravel(),pc.ravel())[0,1]);cm=read(dest/f'conditional_{step}.csv');cc=[r for r in cm if r['arm']==name];rec=[float(r['sign_flip_tp'])/float(r['sign_flip_true']) for r in cc if float(r['sign_flip_true'])>0];conditional.append(dict(seed=seed,step=step,arm=name,pooled_double_center_corr=corr,event_median_corr=float(np.median([float(r['ranking_interaction_corr']) for r in cc if r['ranking_interaction_corr']!='nan'])),event_mean_flip_recall=float(np.mean(rec)) if rec else None))
   for dimension in ['overall','layer','state_role']:
    groups=collections.defaultdict(list)
    for r in rs:groups[(r['arm'],r['policy'],'all' if dimension=='overall' else r['layer'] if dimension=='layer' else str(int(r['state'])%2))].append(r)
    for (arm,policy,value),v in groups.items():
     ss=decision_summary(v);ss.update(seed=seed,step=step,arm=arm,policy=policy,dimension=dimension,value=value)
     for k in ['local_mse','local_mass_mse','beta_mae','beta_contrast_mae']:ss[k]=float(np.mean([float(r[k]) for r in v]))
     summaries.append(ss)
  for name in ['M16B16','M16B128','Q64B16','Q64B128']:
   rr=[r for r in read(dest/'layer29_teachers.csv') if r['method']==name];teacher.append(dict(seed=seed,method=name,**decision_summary(rr)))
  rr=read(dest/'physics.csv')
  for layer in ['1','15','29']:
   v=[r for r in rr if r['layer']==layer];r=dict(seed=seed,layer=layer)
   for k in ['phi_near99_fraction','phi_exact1_fraction','phi_mean_derivative','bs_mean','beta_loss','local_loss','set_loss','quality_output_gradnorm','set_beta_output_gradnorm','common_error_sse_fraction','common_quality_gradient_energy_fraction']:r[k]=float(np.median([float(x[k]) for x in v]))
   physics.append(r)
 table(D/'summary.csv',summaries);table(D/'conditional_summary.csv',conditional);table(D/'physics_summary.csv',physics);table(D/'teacher_summary.csv',teacher)
 lines=['# E44：E43优化公平性、任务损失与Layer29诊断','', '仅使用E43既定48训练事件/96状态。六个分支各2304步，最终检查点固定；无开发/Test拟合或选择。L使用每轨迹/layer/head独立Adam，只更新活动组；每组96步，未访问参数不推进动量。学习率保持1e−4，未使用24倍学习率。稀疏时钟不等同于P/L参数量或更新幅度公平。','', 'P_contrast固定group/contrast系数各1、组内均匀权重：四角色均值Huber加R/A池内中心化Huber，均沿用原bs归一化；这是一个明确固定实现，不是已验证最优设计。P_physical/L_physical仅local+quality，P_set/L_set仅setloss，L_joint_sparse为原三项联合。','', '|seed|分支|Set恢复|H_loss|局部MSE|β MAE|β池内对比MAE|','|---|---|---:|---:|---:|---:|---:|']
 for r in summaries:
  if r['step'] in [-1,2304] and r['dimension']=='overall' and r['policy']=='Set':lines.append('|%s|%s|%.2f%%|%.6f|%.5g|%.5g|%.5g|'%(r['seed'],r['arm'],r['recovery_pct'],r['H_loss'],r['local_mse'],r['beta_mae'],r['beta_contrast_mae']))
 lines+=['','## Layer29真实教师四格','', '|seed|教师|Q64真实收益恢复|H_loss|','|---|---|---:|---:|']
 for r in teacher:lines.append('|%s|%s|%.2f%%|%.6f|'%(r['seed'],r['method'],r['recovery_pct'],r['H_loss']))
 lines+=['','四格均在两个状态/16个事件上同候选池评价。Q64B128评分闭合到完整真实增量。M16或投影教师不是完整目标严格上界。','', '## 条件信息与物理假设','', '|seed|分支|汇总双中心相关|逐事件中位相关|符号翻转召回均值|','|---|---|---:|---:|---:|']
 for r in conditional:
  if r['step'] in [-1,2304]:lines.append('|%s|%s|%.4f|%.4f|%s|'%(r['seed'],r['arm'],r['pooled_double_center_corr'],r['event_median_corr'],r['event_mean_flip_recall']))
 lines+=['','每个scores文件保存96×512完整评分矩阵及真实收益，两个状态使用同候选动作；同时保留每事件指标，汇总相关不能替代事件层变化。符号翻转极少时召回不稳定，不能由相关性推出实际收益。','', '|seed|layer|φ近饱和比例|平均导数中位数|bs均值中位数|质量梯度共模能量比例|','|---|---|---:|---:|---:|---:|']
 for r in physics:lines.append('|%s|%s|%.3f%%|%.4f|%.4f|%.2f%%|'%(r['seed'],r['layer'],100*r['phi_near99_fraction'],r['phi_mean_derivative'],r['bs_mean'],100*r['common_quality_gradient_energy_fraction']))
 lines+=['','共模梯度能量在每Query页面等权欧氏投影下计算，不等于参数更新占比，不能直接称99%训练梯度被吸收。bs与mean reduction均保留，输出梯度与参数梯度分别保存。φ被冻结，饱和只限制输入可辨识性，不存在训练φ梯度消失的诊断结论。','', '## 审计与解释边界','', '已离线读取原KV和token ID重新计算哈希；历史身份匹配E40原哈希，当前页同样记录原KV/ID哈希；所有摘要哈希重新核验。摘要等价但原KV不同的碰撞数必须为0，否则实验停止。组内两tile的页面系数索引完全相同；两个状态共享全部系数，原Medium保持合法并保护own。未重新执行原始Attention。','', '保存E43原P/L最终模型全部评分，补齐旧报告缺失矩阵。所有新增分支初始化动作一致，L各活动组Adam时钟最终均96。训练记录三项loss和裁剪范数；纯物理分支分阶段表可判断误差与收益是否同步，但有限检查点趋势不证明数学单调性。','', '本轮仍是有限优化过程：L具有更多自由参数，不能从收敛差异证明摘要信息不足；纯setloss可能通过响应扭曲改善离散决策，不能把其输出当物理误差估计。比较P_set的响应MSE/β误差与收益，仍不能将全部改进严格因果分摊给两个分支。没有扩大Reader或改变在线结构。','', '文件：summary.csv含所有检查点/layer/状态角色；conditional_summary.csv及每事件conditional表；physics_summary.csv与逐事件梯度；teacher_summary.csv；各种子完整评分、身份表、最终参数和原始曲线。Test保持封存。']
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n');dest=ROOT/'docs/research/evidence/e44_fit_diagnostics_v1';dest.mkdir(parents=True,exist_ok=True)
 for n in ['REPORT.md','summary.csv','conditional_summary.csv','physics_summary.csv','teacher_summary.csv']:shutil.copy2(D/n,dest/n)
 for seed in [17,29]:
  for n in ['protocol.json','complete.json','identity_audit.json','physics.csv','physics_gradients.csv','layer29_teachers.csv','conditional_-1.csv','conditional_2304.csv','evaluation_2304.csv']:shutil.copy2(D/f's{seed}'/n,dest/f's{seed}_{n}')
 with tarfile.open(D/'e44_review.tar.gz','w:gz') as tf:
  for p in D.rglob('*'):
   if p.is_file() and p.suffix in ['.csv','.json','.npz','.md']:tf.add(p,arcname=str(p.relative_to(D)))
 print('REPORT DONE',flush=True)
if __name__=='__main__':run()
