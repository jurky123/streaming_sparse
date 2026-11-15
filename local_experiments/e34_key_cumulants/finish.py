from .report import *
import tarfile

def main():
 summary=json.loads((DEST/'summary.json').read_text());gates=json.loads((DEST/'gates.json').read_text());audit=json.loads((DEST/'independent_audit.json').read_text());quality=list(csv.DictReader((DEST/'quality_summary.csv').open()));conditions=list(csv.DictReader((DEST/'conditional_summary.csv').open()));mom=list(csv.DictReader((DEST/'moment_errors.csv').open()));general=list(csv.DictReader((DEST/'generalization.csv').open()));reuse=json.loads((DEST/'reuse_summary.json').read_text());ss={(r['seed'],r['method'],r['policy']):r for r in summary}
 def f(v,fmt='.4f'):return format(float(v),fmt) if v not in [None,''] else 'NA'
 lines=['# E34：Key分布矩的低成本β估计','', '**判定：'+('LowRank8通过预登记开发筛选，仍未进入部署。' if gates['passed'] else 'LowRank8未通过预登记开发筛选；停止本轮固定矩配置，不扩大rank、不调二阶系数、不训练或接Kernel。')+'**','', '## 范围与固定方法','', '两个冻结C/E29种子，各288个相同开发事件，来自kitchen_cook_s701与snow_ski_s702两条轨迹。Q64/Page64/R4、M16、10%精算预算、own保护、原Risk、Medium16×32、一次交换/no-op全部固定。所有方法仅改变供集合组合使用的β，不改变近似Attention或初选。当前与历史页都使用相同解析规则。Test未访问，没有新QKV或神经网络训练。','', '|方法|均值项|方差项|信息地位|','|---|---|---|---|','|E29|原Reader|原Reader|原在线基线|','|Variance-only|0|原stats block8|零新增统计|','|Full-Mean+Variance|真实完整均值偏移|原stats block8|完整均值容量诊断|','|LowRank8+Variance|训练基投影，32个BF16坐标/页|原stats block8|64 B/page候选|','|Exact-Q-Cumulant2|真实完整均值|原始Key的真实Query方向总体方差|离线二阶参考|','|TrueBeta|精确页面logZ差|精确|教师参考|','', 'beta=LSE(ell+mu+variance/2)−LSE(ell)，系数1/2固定；无拟合校准、裁剪或根据结果选阈值。这里是log矩生成函数的二阶累积量截断，不是E19对exp的二阶Taylor多项式。两个近似不能混称为同一个M2。','', '本轮替换所有页β，并重建全部集合和全页面预测和。Set与Independent使用完全相同的信息、候选池和预算。独立策略按预测页面净残差能量选一次合法交换或no-op。各方案的真实N/Z和Q64误差标签不变，因为R4及最终混合计算路径没有变化。','', '## 训练基与跨Query复用','', '每个模型种子单独固定12个layer/head基，每基8维，在对应layer/head所有页面之间共享。只使用原E24八条训练轨迹，按轨迹内有序token IDs＋原始K＋summaryK去重；四子组均值偏移等权，拟合未中心化PCA8。没有使用开发Query、残差或收益选基；这种L2最优基不保证Query加权或决策最优，失败不能推广为任意rank8空间都失败。','', '基存FP32，页面32个坐标存BF16，再用Query投影读出均值项。各方案的解码、累积量和logsumexp均用FP64参考计算，不能视为已验证的部署FP32路径。页面系数不按当前Query重新拟合，也不读取其他上下文标签；重复历史页的系数与原方差统计逐字节一致。','', '|seed|训练物理页面状态数（12基合计）|PCA8能量保留均值|开发历史页出现|重复出现|','|---|---:|---:|---:|---:|']
 for seed in [17,29]:
  lock=json.loads((DEST/f's{seed}/basis_lock.json').read_text());r=next(x for x in reuse if x['seed']==seed);lines.append(f"|{seed}|{sum(x['unique_pages'] for x in lock['audit'])}|{np.mean([x['energy8'] for x in lock['audit']]):.2%}|{r['history_occurrences']}|{r['repeated_occurrences']}|")
 lines+=['','这是可复用性的一致性检查，不等于跨Query准确性已通过。reuse-tile/step/forward误差与支持质量覆盖见quality_summary.csv和recurrence_coverage.csv；这些行描述重复出现页面上的误差，不是再次拟合LOTO系数的实验。','', '## 实际组合决策收益','', 'Oracle恢复=Σ真实执行收益/Σ同池真实最佳正收益。下表为M16评分、Q64真实评价；正相对改善表示更好。事后真门控读取标签，只用于诊断动作提议，不属于线上方法。','', '|方法|seed17 Set恢复|seed29 Set恢复|seed17汇总NMSE改善|seed29汇总NMSE改善|','|---|---:|---:|---:|---:|']
 for method in METHODS:
  a,b=[ss[s,method,'Set'] for s in [17,29]];lines.append(f"|{method}|{a['R_executed']:.2%}|{b['R_executed']:.2%}|{a['pooled_improvement_pct']:.3f}%|{b['pooled_improvement_pct']:.3f}%|")
 lines+=['','|seed|方法|Set恢复|Independent恢复|提议＋真门控恢复|逐例平均改善|退化率|no-op|P10|最差相对改善|最大绝对NMSE增加|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for seed in [17,29]:
  for method in METHODS:
   x=ss[seed,method,'Set'];ind=ss[seed,method,'Independent'];lines.append(f"|{seed}|{method}|{x['R_executed']:.2%}|{ind['R_executed']:.2%}|{x['R_proposal']:.2%}|{x['mean_improvement_pct']:.3f}%|{x['degraded']/288:.2%}|{x['noop']}|{x['p10_pct']:.3f}%|{x['worst_pct']:.3f}%|{x['max_absolute_nmse_increase']:.6g}|")
 lines+=['','相对指标仅对初始J>1e−12定义，绝对汇总保留所有事件；改善/退化计数阈值1e−12。极小初始误差可放大相对尾部，故并列绝对NMSE增加，完整final NMSE与regret见set_decision.csv。两条轨迹内事件相关，288不是288条独立视频。','', '## 质量误差与近似来源','', '下表按事件等权平均（M16、全页）。质量加权MAE在每事件内部用真实页面质量归一化。相对Z误差可能超过浮点表示范围，CSV另存log误差和有限值数量；NA不能按零误差处理，未删掉对应事件或实际动作。','', '|seed|方法|β MAE|β有符号偏差|真实质量加权β MAE|相对Z L1的log均值|','|---|---|---:|---:|---:|---:|']
 for r in quality:
  if r['queries']=='M16' and r['region']=='all':lines.append(f"|{r['seed']}|{r['method']}|{f(r['beta_mae'])}|{f(r['beta_bias'])}|{f(r['true_weighted_beta_mae'])}|{f(r['quality_mass_log_l1_relative'])}|")
 lines+=['','TrueBeta的log相对Z误差为NA是因为真实误差严格为0、log(0)不作为有限数写出；原始L1/L2为0。其余超范围情况通过有限值数量和log指标保留，不混同为零。','','完整均值满足 ell+mu=q·k_mean/√128，所以Full-Mean和Exact-Q-Cumulant2重建的页面质量不再依赖R4 Key位置。同一原始QK在两个C种子下的未加权logZ误差一致是解析恒等式的结果，不是复制结果；它们的集合决策仍因预测响应、摘要中心和初选不同而不同。','','β有符号偏差=估计−真值。二阶参考即便使用真实方向均值/方差，仍只是截断近似，不是真实β。较大的正偏差说明不能把方差项指数化后默认得到可靠质量；动作质量仍须单独验证。对只有16个Key的离散组，精确log-mean-exp受最大logit偏移约束，二阶累积量项则随logit振幅的平方增长（对方差为线性），较大logit波动下没有一般准确性保证。','', '|seed|Query方向均值投影相对MSE（事件均值）|分块方差相对MSE（事件均值）|完整二阶β偏差（Q64事件均值）|','|---|---:|---:|---:|']
 for seed in [17,29]:
  rr=[r for r in mom if int(r['seed'])==seed];vals=[np.mean([float(r[k]) for r in rr]) for k in ['mean_projection_relative_mse','variance_relative_mse','full_c2_beta_bias']];lines.append('|'+str(seed)+'|'+'|'.join(f'{v:.5g}' for v in vals)+'|')
 lines+=['','高估并非均匀发生：完整二阶参考的Q64平均β偏差在layer1约0.951、layer15约0.307，而layer29约2034.356。需要结合分层决策评价，不能将汇总偏差解释成所有层都同样失败。','','|seed|层|E29恢复|LowRank8恢复|完整二阶参考恢复|','|---|---:|---:|---:|---:|']
 for seed in [17,29]:
  for layer in [1,15,29]:
   vals=[next(float(r['R_executed']) for r in general if int(r['seed'])==seed and r['dimension']=='layer' and int(r['value'])==layer and r['method']==method) for method in ['E29','LowRank8+Variance','Exact-Q-Cumulant2']];lines.append('|'+str(seed)+'|'+str(layer)+'|'+'|'.join(f'{v:.2%}' for v in vals)+'|')
 lines+=['','均值PCA能量保留与Query方向预测误差不是同一指标；完整二阶参考和完整均值参考之间的差距也可能由误差抵消造成，不能假设更多真实矩信息必然单调改善动作排序。','', '## 集合条件与轨迹分层','', '|seed|方法|双重中心化相关|逐事件交互相关中位数|符号翻转召回|','|---|---|---:|---:|---:|']
 for r in conditions:lines.append(f"|{r['seed']}|{r['method']}|{f(r['doublecentered'])}|{f(r['median_event_interaction'])}|{f(r['sign_flip_recall'],'.2%')}|")
 lines+=['','沿用五状态共同合法面板。预测评分除以真实Dense正能量仅作离线量纲比较，不进入β/Reader或动作/no-op；独立审计检查去除该正常数不改变接受判断。汇总相关受跨事件尺度影响，因此不能只选相关性最高的一种口径。','', '|seed|方法|轨迹|汇总NMSE改善|Oracle恢复|','|---|---|---|---:|---:|']
 for r in general:
  if r['dimension']=='trajectory':lines.append(f"|{r['seed']}|{r['method']}|{r['value']}|{float(r['pooled_improvement_pct']):.3f}%|{float(r['R_executed']):.2%}|")
 lines+=['','## 成本账','', 'LowRank8新增每页4×8×BF16=64 B；338历史页×12 heads×29层=7.18 MiB，72当前页若缓存另为1.53 MiB。推广到29层的共享FP32基约1.36 MiB，当前实验实际只使用3层×4heads。','', '稳态每层12heads×72tiles×16probes×410页：Query投影约0.028 GFLOPs，均值读出约0.363 GFLOPs。若不能复用Risk已计算的分块方差，还需约0.363 GFLOPs。另有每Query–Page四项指数与logsumexp、归一化、摘要读取及原Reader/集合评分；这些未折算为FLOPs或实测时间。无跨tile复用时新增64 B坐标的逻辑读取约21.62 MiB/层。','', '当前KV每次刷新仍须构建系数：复用现有摘要构建已算出的子组Key均值，额外投影及减法约0.0075 GFLOPs/层/刷新。历史系数在clean提交时构建一次；不能把当前页刷新成本省略。Full-Mean需要1024 B/page均值向量，Exact-Q-Cumulant2直接用原始K，仅为离线参考，均不能继承64 B候选的成本。','', '没有新原始KV探测；离线规范KV用于模拟提交时统计和教师诊断。上述只是逻辑读写与主乘加，不是Kernel延迟或加速比。','', '## 门槛与停止','']
 for g in gates['results']:
  lines.append(f"- seed{g['seed']}："+('通过。' if g['passed'] else '未通过；'+', '.join(k for k,v in g['checks'].items() if not v)+'。')+f" E29→TrueBeta增益缺口恢复 {g['true_beta_gap_recovery']:.2%}。")
 lines+=['','预登记要求两个种子均≥20%Oracle恢复、两开发轨迹正收益、Set同时优于E29与同信息Independent、逐例均值非负、退化≤15%、P10与最差不劣于E29。无事后降门槛，未访问Test。','', '结果只判断本轮固定PCA8＋分块方差＋二阶累积量结构。不通过时停止本配置，不据此断言所有小型辅助统计都不可能有效；也不把完整二阶参考的正汇总收益当作可靠部署证明。真实β本身仍有剩余退化，说明局部响应与接受判断的限制依然存在。','', '## 审计与文件','', f"全部{sum(r['decisions'] for r in audit['counts']):,}个动作通过count/own/候选合法性检查；{len(audit['checks'])}个选中动作由NumPy longdouble独立重算真实输出与Set评分；{len(audit['beta_checks'])}组β公式复算；24个训练基谱/正交性核验及{audit['train_kv_files_verified']}份训练KV哈希核验通过。模型、基和协议SHA保持不变；重复历史页坐标一致。",'', '主要表格：summary.csv、set_decision.csv、quality_summary.csv、moment_errors.csv、conditional_summary.csv、generalization.csv、cost_estimate.csv。原始beta/共同面板保存在s17/s29；basis.pt及basis_lock.json保存固定训练基与来源。精简包不含原始QKV、模型或大型逐事件缓存，保留报告、CSV、固定基、协议与审计。']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
 entries={p.name:p for p in DEST.glob('*.csv')}
 for n in ['REPORT.md','protocol.json','gates.json','summary.json','independent_audit.json','reuse_summary.json']:entries[n]=DEST/n
 for seed in [17,29]:
  for n in ['complete.json','basis_lock.json','basis.pt']:entries[f's{seed}/'+n]=DEST/f's{seed}'/n
 for p in (ROOT/'local_experiments/e34_key_cumulants').glob('*'):
  if p.is_file():entries['source/'+p.name]=p
 inventory={n:sha(p) for n,p in entries.items()};dump(DEST/'review_inventory.json',inventory);entries['review_inventory.json']=DEST/'review_inventory.json'
 with tarfile.open(DEST/'e34_review.tar','w') as tar:
  for n,p in entries.items():tar.add(p,arcname=n)
 with tarfile.open(DEST/'e34_review.tar') as tar:
  for n,h in inventory.items():assert hashlib.sha256(tar.extractfile(n).read()).hexdigest()==h
 dump(DEST/'package_audit.json',dict(files=len(entries),bytes=(DEST/'e34_review.tar').stat().st_size,sha256=sha(DEST/'e34_review.tar'),all_inventory_hashes_verified=True));print('FINISHED',len(entries),gates['passed'])
if __name__=='__main__':main()
