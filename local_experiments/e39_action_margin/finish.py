from .report import *

def md(rows,keys,labels=None):
 labels=labels or keys;lines=['|'+'|'.join(labels)+'|','|'+'|'.join(['---']*len(keys))+'|']
 for r in rows:
  vals=[]
  for k in keys:
   v=r.get(k);vals.append('—' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v))
  lines.append('|'+'|'.join(vals)+'|')
 return '\n'.join(lines)
def main():
 assert json.loads((DEST/'audit.json').read_text())['passed'];gates=json.loads((DEST/'gates.json').read_text());rows=list(csv.DictReader((DEST/'decision_summary.csv').open()));dec=list(csv.DictReader((DEST/'decomposition_summary.csv').open()));cc=list(csv.DictReader((DEST/'conditional_summary.csv').open()));primary=[];ind=[];epochs=[];tr=[]
 def cv(r):
  z=dict(r)
  for k,v in z.items():
   try:z[k]=float(v)
   except (ValueError,TypeError):pass
  return z
 for r0 in rows:
  r=cv(r0)
  if r['dimension']!='ALL':continue
  if r['split']=='development':
   if r['epoch'] in [-1,1]:
    r['recovery_pct']=100*r['R_executed'];r['top8_pct']=100*r['R_top8'] if r.get('R_top8') not in ['',None] else None
    if r['policy']=='Set':primary.append(r)
    else:ind.append(r)
   if r['policy']=='Set':epochs.append(r)
  elif r['policy']=='Set':tr.append(r)
 d=[cv(x) for x in dec if x['dimension']=='ALL' and x['subset']=='different'];c=[cv(x) for x in cc if x['epoch'] in ['1','-1']]
 mech='通过' if gates['mechanism_pass'] else '未通过';usable='通过' if gates['online_pass'] else '未通过'
 text='''# E39：关键动作对比续训与同动作对误差分解

**A分解与T0/T1两种子续训已完成。** 不改变原在线结构、摘要、Reader容量、M16、Medium或Risk。每组2304次更新，固定最终epoch1为主结果；没有按开发表现挑检查点，没有追加训练、门控或超参数搜索。Test保持封存。

## 执行边界

两种子分别从各自E29 selected模型开始；两个训练组权重相同，Adam均重置（旧检查点没有优化器状态）。C摘要、Adaptive φ、B16固定，Reader仍为51,157参数。原8条训练轨迹、2条开发轨迹，开发每种子288个事件；这不是576条独立视频。状态、顺序、标签与优化器一致。两组原局部响应与质量监督保留，T1只将gainHuber+0.1listwise替换为Top4竞争集合的cost-augmented max margin。

A复用E38/E30缓存，没有重算原始Attention。续训离线读取已有训练KV重建冻结摘要，真实N/Z使用已有缓存。线上没有新增KV读取、缓存、模块或接受阈值。训练保持原E29数值路径，开发使用等价的分别归一化E30路径，独立审计检验一致性。

## A：同一动作对的闭合分解

以下只列原模型与真Oracle动作不同的事件；完整数据另含全部事件、原有害提议及分层统计。

'''
 text+=md(d,['seed','n','true16_correct','true64_correct','prediction_error_sum','query_difference_sum','projection_difference_sum','prediction_error_abs_sum','query_difference_abs_sum','projection_difference_abs_sum'])
 text+='''

三项分别为预测M16/B16−真实M16/B16、真实M16/B16−真实Q64/B16、真实Q64/B16−真实Q64/full，严格闭合。保留符号，绝对和另报。多数固定动作对在真实M16/B16下已经可正确区分；这证明评分空间内存在可区分性，不证明这些真值可从R4预测。较大的有符号投影总量并不否定这一点，也不能解释为所有执行损失的百分比归因。

M16与Q64均使用原冻结B16坐标，保持与E38 M11一致；不是重新拟合或更换空间。B在有限精度下非严格正交，因此“投影差异”是这一实际坐标评分目标与全维目标的差异，不强行当成精确正交能量归因。

## 最终检查点：实际Set收益与损害

'''
 text+=md(primary,['seed','group','recovery_pct','executed_gain_sum','H_gain','H_loss','mean_improvement_pct','degraded','noop','p10_pct','worst_absolute_increase','top8_pct'],['seed','组','Oracle恢复%','净收益和','H_gain','H_loss','逐例均值%','退化数','no-op','P10%','最差绝对增量','Top8覆盖%'])
 text+='''

恢复率分母为相同Risk和Medium面板真实Q64最佳收益之和。Top-L使用真实标签事后选优，只是召回参考。H_gain/H_loss分列，净收益=H_gain−H_loss。相对改善只计初始NMSE>1e−12事件，样本数在CSV中明确保存。

## 同信息Independent

'''+md(ind,['seed','group','recovery_pct','executed_gain_sum','H_gain','H_loss','degraded'])
 text+='''

Independent与Set使用同一模型输出、初选、候选和预算；不能因T1比旧E29更好就忽略匹配预算T0。

## 固定旧动作对与错误转移

'''+md([r for r in primary if r['group']!='E29'],['seed','group','fixed_margin_abs_error','fixed_preference_correct','old_harmful','old_rejected','old_rejected_new_harmful'])
 text+='''

最后一列只计原有害动作被当前模型降为非正后，重新决策仍有害的事件。旧动作面板与新提议同时保留，避免把压低原错误动作当成整体修复。跨五状态只在旧动作对双方均合法的共同面板上评价；不合法或不在共同面板的动作不强行计入，覆盖数见fixed_pair_conditional.csv。

## 跨集合条件信息

'''+md(c,['seed','group','double_centered_corr','median_event_corr','sign_flip_n','sign_flip_recalled','sign_flip_recall'])
 text+='''

双重中心化相关使用共同合法动作的5状态收益。符号翻转召回剔除任一真实状态收益绝对值≤1e−12的对，不用总体正确率代替少数翻转事件召回。

## 两个预定检查点全部披露

'''+md(epochs,['seed','group','epoch','R_executed','H_gain','H_loss','degraded','p10_pct'])
 text+='''

主比较始终使用epoch1，epoch0仅展示过程，不能事后改选更好的一轮。

## 固定训练监测面板

'''+md(tr,['seed','group','epoch','R_executed','H_loss','degraded'])
 text+='\n\n'+md([cv(x) for x in csv.DictReader((DEST/'train_monitor_summary.csv').open())],['seed','group','epoch','n','mean_margin_loss','median_margin_loss'])
 text+='''

训练监测为预先固定的48个事件（排序后每24个取1），以Risk状态评估；它不代表完整训练集和所有训练反事实状态。训练过程中每次更新的local/quality/margin/set损失另存training_curves.csv。T0与T1的总loss不是同一目标，不能仅比较总loss高低。信息回归指标不用于选择最终模型。

## 判定与后续

'''
 text+=f'预登记训练机制门槛：**{mech}**。部署门槛：**{usable}**。逐种子条件见gates.json。\n\n'
 if not gates['mechanism_pass']:
  text+='本轮未能在两个种子上同时证明关键动作对比损失比匹配预算续训获得更高实际收益且减少损害。停止对同一Reader追加hard-negative数量、损失权重或训练时长搜索。不由此推出R4信息论上不足；训练优化、目标近似与跨轨迹泛化仍是不同问题。\n'
 elif not gates['online_pass']:
  text+='关键动作监督通过了有限训练机制门槛，但可靠性和同信息组合优势等部署要求尚未全部通过；只能报告优化方向有效，不能将它称为可靠在线修正器，也不解封Test。\n'
 else:text+='开发门槛通过，但仅有两条开发轨迹，不等同端到端速度或泛化验证；本轮不自动访问Test。后续应保持模型与规则冻结，再安排一次独立测试。\n'
 text+='''

本轮的具体限制：T1固定训练监测面板平均margin从1.814降至1.674（seed17）、1.782降至1.746（seed29）；有一定优化响应，但不能称为训练完全解决。最终旧动作对偏好正确计数较T0从25/28升到30/36，Top8覆盖也小幅增加；它们没有转化成两个种子一致更高的实际净收益。seed29相对T0的增益还应与训练前E29比较，避免把T0续训退化误作T1的全部新增能力。T1退化数98/99均高于T0的93/95。条件相关、翻转召回也没有一致改善。因此这不是“已解决训练、只剩泛化”的证据，更不能归纳为“R4没有信息”。

## 审计和复现

- 固定协议与源码hash在protocol.json；完整训练/状态顺序在schedule_s17/s29.json，两个组共享同一个顺序。
- 检查初始化与E29相等、C/φ冻结、2304更新、count/own和相同候选面板。
- 独立NumPy longdouble复核最终模型抽样收益和A的M16/Q64间隔；独立重算两个组的第1步目标和hard集合。
- 三项分解逐事件闭合；预检覆盖无正收益动作/no-op、竞争集合梯度与旧新数值评分等价。
- 审计结果见audit.json。未新增Kernel或延迟测量，不能把相同推理结构解释为已测得加速。

关键表：decomposition_summary.csv、decision_summary.csv、decision_per_event.csv、conditional_summary.csv、fixed_pair_conditional.csv、training_curves.csv、information_quality.csv。完整模型、优化器与逐事件数组保留在本地artifacts。评审包不包含原始QKV或模型权重。
'''
 (DEST/'REPORT.md').write_text(text)
 dump(DEST/'stage_status.json',dict(A='complete',training='complete_2304_each_arm_seed',main_checkpoint='epoch1',test='sealed',mechanism_pass=gates['mechanism_pass'],online_pass=gates['online_pass']))
 source=list((ROOT/'local_experiments/e39_action_margin').glob('*.py'))+[ROOT/'local_experiments/e39_action_margin/README.md'];items=[p for p in DEST.iterdir() if p.suffix in ['.csv','.json','.md'] and p.name!='package_manifest.json']+source;manifest=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in items];dump(DEST/'package_manifest.json',manifest)
 with tarfile.open(DEST/'e39_review.tar.gz','w:gz') as tar:
  for p in items+[DEST/'package_manifest.json']:tar.add(p,arcname=str(p.relative_to(ROOT)))
 with tarfile.open(DEST/'e39_review.tar.gz') as tar:
  for x in manifest:assert hashlib.sha256(tar.extractfile(x['path']).read()).hexdigest()==x['sha256']
 print('FINISHED',DEST,flush=True)
if __name__=='__main__':main()
