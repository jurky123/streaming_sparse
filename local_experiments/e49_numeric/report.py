from .run import *
from ..e41_training_signal.report import decision_summary
import csv,shutil,tarfile

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def run():
 decisions=[];paths0=[];refs=[];rejected=[]
 for seed in [17,29]:
  for part in range(4):
   dest=D/f's{seed}_part{part}';assert (dest/'complete.json').exists();decisions+=read(dest/'decisions.csv');paths0+=read(dest/'path_checks.csv')
  for kind in ['tails','all']:refs+=read(D/f'reference_{seed}_{kind}.csv')
 summaries=[]
 for dimension in ['overall','layer','head','trajectory']:
  groups=collections.defaultdict(list)
  for r in decisions:groups[(r['seed'],r['split'],r['model'],r['policy'],r['protocol'],'all' if dimension=='overall' else r[dimension])].append(r)
  for (seed,split,model,policy,protocol,value),rs in groups.items():
   ss=decision_summary(rs);loss=sorted([max(0.,-float(r['executed_gain'])) for r in rs]);ss.update(harm_lt_1e_minus3=sum(-1e-3<float(r['executed_gain'])< -1e-12 for r in rs),damage_p90=float(np.percentile(loss,90)),damage_p99=float(np.percentile(loss,99)));summaries.append(dict(seed=seed,split=split,model=model,policy=policy,protocol=protocol,dimension=dimension,value=value,**ss))
 groups=collections.defaultdict(list)
 for r in decisions:
  if r['protocol']=='interval' and r['execute']=='False' and float(r['raw_score'])>0:groups[(r['seed'],r['split'],r['model'],r['policy'])].append(r)
 for key,rs in groups.items():rejected.append(dict(zip(['seed','split','model','policy'],key),count=len(rs),real_positive=sum(float(r['proposal_gain'])>1e-12 for r in rs),real_negative=sum(float(r['proposal_gain'])< -1e-12 for r in rs),forgone_gain=sum(max(0.,float(r['proposal_gain'])) for r in rs),avoided_harm=sum(max(0.,-float(r['proposal_gain'])) for r in rs)))
 table(D/'summary.csv',summaries)
 if not rejected:
  rejected=[dict(seed=seed,split=split,model=model,policy=policy,count=0,real_positive=0,real_negative=0,forgone_gain=0.,avoided_harm=0.) for seed in [17,29] for split in ['train','development'] for model in ['E29','T0','Full_Pset'] for policy in ['Set','Independent']]
 table(D/'rejected.csv',rejected)
 lines=['# E49：冻结评分数值决策审计','', '固定E29、同预算T0、E48 Full_Pset；各1152训练/288开发事件，两种子共8640模型事件。无训练、无摘要/候选变化，Test封存。历史全部评分和选择保留；统一/增量路径仅作为数值对照，区间规则只核验历史最高提议，不改排名、不试第二名。','', '## 数值规则','', 'Reader实际BF16/FP32输出与传入评分的lm/center二进制值视为精确输入。Set使用未除Dense能量的原始gain；Independent使用一个固定正缩放下的等价能量差，仅符号可与旧能量比较。指数用MPFR160位有方向舍入，基本加减乘除逐运算nextafter外向扩张；除法包含0或非有限路径给出无界区间并拒绝。区间下界>0才执行，否则no-op。这是对固定代理的数值符号核验，不保证真实收益正，也不包含Reader建模误差。','', '保证范围为本机IEEE754 binary64正常/非正规数与最近舍入、无fast-math、规定算术表达式。MPFR exp本身定向舍入，避免假定GPU exp误差只有一个ULP。非部署硬件通用保证。统一FP64归约与端点增量保留正项新分母；移出主导项导致端点差不稳时回退统一路径，不宣称更快。','', '## 全量实际收益','', '|seed|划分|模型|协议|Set恢复|H_gain|H_loss|退化|最大绝对损害|','|---|---|---|---|---:|---:|---:|---:|---:|']
 for r in summaries:
  if r['dimension']=='overall' and r['policy']=='Set' and r['protocol'] in ['original','interval']:lines.append('|%s|%s|%s|%s|%.2f%%|%.6f|%.6f|%d/%d|%.6f|'%(r['seed'],r['split'],r['model'],r['protocol'],r['recovery_pct'],r['H_gain'],r['H_loss'],r['harm'],r['events'],r['worst_absolute_increase']))
 lines+=['','同信息Independent、统一归约/增量路径的实际收益，以及layer/head/trajectory分列均见summary.csv。rejected.csv保留被拒动作的真实正/负数量与损害/收益，不以真实标签选接受规则。真实统计容差仍沿用1e−12；此容差不参与接受。','', '## 高精度参考','', '按已知灾难、最小正预测分数、预定事件顺序正控制选样；不根据真实gain优化阈值。精度按端点质量跨度预定并增加80位复算，检查符号稳定及区间包含。极小或有限精度零不能宣称绝对精确零。完整十进制结果在reference表，不用float重新舍入它来证明符号。','', '|seed|模型|样本|原始FP64 raw|高精度gain（截取）|真实gain|','|---|---|---|---:|---|---:|']
 for r in refs:
  if r['selected']=='True':lines.append('|%s|%s|%s|%s|%s|%.6g|'%(r['seed'],r['model'],r['scope'],r['raw_fp64'],r['mp_gain'][:70],float(r['real_gain'])))
 lines+=['','两个已知灾难中旧E29的原始代理收益高精度仍为正且区间下界为正，不支持“一个ULP噪声误触发”的解释。旧归一化分数很小，但不能把归一化值直接拿来与1附近epsilon比较。T0和Full_Pset在seed29尾部会给负分而拒绝；seed17仍可数值可靠地误判为正。数值可靠性不解决页面响应/质量预测错误。','', '所有512评分、no-op字面0、最高与次高间隔、路径最高动作均保存；端点β/页面质量见endpoint表，输出变化及E范数见output_change_reference.csv（FP64范数可能因取消为0，不当高精度零证明）。','', '## 范围与后续','', '此规则只有符号可分辨性含义，不是实际收益阈值。没有改成Dense归一化后的1e−12，也没有根据损害调宽区间。未被区间排除的真实负收益应归模型错误；本轮不据此启用温度/范数补偿、摘要训练或接受策略搜索。整体收益与最差尾部须同时判断，不能只报退化数下降。']
 (D/'REPORT.md').write_text('\n'.join(lines)+'\n');out=ROOT/'docs/research/evidence/e49_numeric_v1';out.mkdir(parents=True,exist_ok=True)
 for n in ['REPORT.md','summary.csv','rejected.csv','audit.json','interval_self_checks.csv','output_change_reference.csv']:
  if (D/n).exists():shutil.copy2(D/n,out/n)
 for seed in [17,29]:
  for kind in ['tails','all']:
   for stem in ['reference','endpoint']:shutil.copy2(D/f'{stem}_{seed}_{kind}.csv',out/f'{stem}_{seed}_{kind}.csv')
 print('REPORT DONE',flush=True)
if __name__=='__main__':run()
