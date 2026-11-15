from .core import *
import shutil,tarfile,ast

def run():
 s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'audit.json').read_text());lock=json.loads((OUT/'calibration_lock.json').read_text());assert audit['passed']
 # Explicit tails and trajectories: no clipping or removal of difficult states.
 chosen=list(csv.DictReader((OUT/'C_selected_actions.csv').open()));state_rows=[];tails=[]
 for sp,method,state in sorted({(r['split'],r['method'],r['state']) for r in chosen}):
  rr=[r for r in chosen if (r['split'],r['method'],r['state'])==(sp,method,state)];gain=np.array([float(r['gain']) if r['execute']=='True' else 0 for r in rr]);J=np.array([float(r['nmse']) for r in rr]);rel=gain/J*100
  state_rows.append(dict(split=sp,method=method,state=state,n=len(rr),mean_relative_pct=float(rel.mean()),median_relative_pct=float(np.median(rel)),p10_relative_pct=float(np.quantile(rel,.1)),min_relative_pct=float(rel.min()),mean_absolute_gain=float(gain.mean()),degraded=int(np.sum(gain< -1e-12))))
  if sp=='evaluation':
   for j in np.argsort(rel)[:3]:tails.append(dict(**rr[j],relative_pct=float(rel[j])))
 table('C_by_state.csv',state_rows);table('C_worst_actions.csv',tails)
 # Add an explicit execution flag to source-action records; prior source groups describe argmax, not gates.
 sources=list(csv.DictReader((OUT/'B_selected_sources.csv').open()));cache={}
 for r in sources:
  if r['event_id'] not in cache:cache[r['event_id']]=np.load(OUT/'events'/r['event_id']/'terms.npz')
  d=cache[r['event_id']];i=int(r['state_index']);j=int(r['action']);sel=r['selection']
  r['execute']=float(r['true_gain'])>0 if sel=='oracle_best' else bool((d[sel.replace('E15_','')+'_unit_I']-d[sel.replace('E15_','')+'_unit_B'])[i,j]>0)
 table('B_selected_sources.csv',sources)
 A=[r for r in s['A'] if r['split']=='evaluation'];B=[r for r in s['B'] if r['split']=='evaluation'];C={(r['scope'],r['method']):r for r in s['C'] if r['split']=='evaluation'};D=[r for r in s['D'] if r['split']=='evaluation']
 at='\n'.join(f"| {r['scope']} | {r['events']} | {r['pooled_V_over_total']:.3f} | {r['pooled_Z_over_total']:.3f} | {r['pooled_cross_over_total']:.3f} |" for r in A if r['level']=='page')
 dt='\n'.join(f"| {level} | "+' | '.join(f"{next(r['cosine'] for r in D if r['level']==level and r['scope']==scope and r['component']==c):.4f}" for c in ['total','V','Z'])+' |' for level,scope in [('page','risk_unselected'),('set','risk'),('candidate','risk')])
 methods=['full_v__proxy_calibrated','full_v__teacherI_Q64','full_v__teacherB_Q64','teacherIB_Q64','full_oracle']
 ct='\n'.join(f"| {m} | {C['risk',m]['gated_relative_pct']:+.3f}% | {100*C['risk',m]['gated_degradation_rate']:.2f}% | {100*C['risk',m]['pooled_gated_recovery']:.2f}% |" for m in methods)
 st='\n'.join(f"| {r['state']} | {r['n']} | {r['mean_relative_pct']:+.3f}% | {r['median_relative_pct']:+.3f}% | {r['degraded']} |" for r in state_rows if r['split']=='evaluation' and r['method']=='teacherIB_Q64')
 calibration='\n'.join(f"| {m['representation']} | {m['E17_lambda']:.6f} | {m['common_scale']:.9f} | {m['train_objective']:.6f} |" for m in lock['models'])
 text=f'''# E18：页面残差来源、交互来源与受控教师替换

**完成144个冻结head–tile、994个集合及457,792个状态–候选动作。没有新视频、QKV采集、Router训练、摘要结构修改、原始KV交叉协方差计算或全局求解。** 统一Q64/Page64、R4/rank8、固定10%预算、own保护。主评价为horse_field两seed的72例，训练city_walk seed10136例仅用于单位校准，开发seed30336例仅分列检查。

## 主要结论

1. **Value响应失真与质量失真都重要，而且强烈抵消。** 不能把总残差按两个分量范数简单分摊成“多少由V、多少由K造成”。初始Risk集合的平均误差能量分解为0.070126＋0.052529−0.112380＝0.010275 NMSE。
2. **Full-V方向代理主要观察到质量失真方向，几乎没有观察到正确的Value响应误差方向。** 初始Risk候选方向与质量分量变化的余弦0.807，与Value分量变化−0.075。高质量分量方向相关性不等于能预测其幅度d。
3. **只把交互项换成真实值不保证改善。** 保留失准的自身能量估计会产生危险动作；部分教师替换不是严格收益上界。初始Risk同时知道真实I、B而不提供D时，恢复面板Oracle汇总收益84.03%，平均相对改善4.889%；完整Oracle为5.437%。
4. **分母重要性依赖所选动作与状态。** 忽略D在初始Risk多数情况有效，但在一个人工扰动状态造成严重退化，不能从E17原评分的错误比例推出D始终次要。

## A：页面残差来源

重建完整64 Query的摘要页输出Ohat_p，采用已有Dense分母归一化数据：

- eZ = d·(Ohat_p−OD)
- eV = e−eZ
- ||e||² = ||eV||²＋||eZ||²＋2eV·eZ

eV描述页内Query-dependent Value响应失真，**不是纯V压缩误差**；它也依赖Key如何改变页内attention权重。eZ是页面质量估计失真。两者的划分是精确恒等式，不是两个独立物理因果干预。

摘要输出用两条独立路线重建（先重建代表V与先算Haar对比系数），并验证 Σ_p e_p = Σ_p Zhat_p/ZD·(OD−Ohat_p)。后者将重建输出与原始冻结贡献独立联系起来，最大差 {audit['max_summary_global_identity_error']:.3e}；不是只验证按定义成立的e=eV+eZ。

### 页面能量：保留负交叉项

下表先汇总每事件能量，再对评价事件等权求均值，最后除以对应总残差能量。三个数相加为1，超过1或交叉项为负都合法；它们不是百分比归因。

| 页面范围 | 有效事件 | V能量/总能量 | Z能量/总能量 | 交叉项/总能量 |
|---|---:|---:|---:|---:|
{at}

“有利交换页面”是初始Risk共同面板中真实收益>1e−12的动作涉及页面的去重并集；最佳交换仅取其两个页面。无有利动作的7例不进入这两组，并保留有效事件数。全页组包含已被Risk选中精算的页面，它们当前不贡献混合近似误差，不能用全页结果替代未选页结论。

### 集合能量：使用真实混合分母

在Risk未选集合内先分别求EV、EZ，再逐Query使用实际Z和统一C加权：

| 集合能量分量 | 评价集平均NMSE单位 |
|---|---:|
| ||EV||²/Z²/C | 0.070126 |
| ||EZ||²/Z²/C | 0.052529 |
| 2EV·EZ/Z²/C | −0.112380 |
| 总实际NMSE | **0.010275** |

逐事件Z占两分量正能量之和的平均为51.06%，中位数44.81%；逐事件总能量/(V能量+Z能量)均值46.16%、中位数49.10%。汇总能量中的抵消更强，说明高能量事件影响明显。完整均值、逐事件数据均保留，不仅报告一个汇总比例。

因此当前证据不支持“只修V”或“只修质量”的单一归因。消去某个分量也不等于真实可实现的摘要修改，因为修改表示可能同时改变两项及其抵消。

## B：交互项的四个来源

I=I_VV＋I_VZ＋I_ZV＋I_ZZ，各项为−2 E来源·δe来源 /Z²/C，并逐Query求和。全部候选四项和与E17真实I最大差 {audit['max_interaction_decomposition_error']:.3e}。

初始Risk全部候选，逐候选四项绝对值归一化、再事件等权：

| 项 | 绝对份额 |
|---|---:|
| VV | 27.90% |
| VZ | 16.13% |
| ZV | 22.36% |
| ZZ | 33.62% |

**这不是四项对最终收益的百分比贡献。** 交叉项同样显著且可正可负。只用VV的符号与完整I相符约62.49%；只用ZZ约67.91%，任一单项都不能稳定代表总交互。真实有利动作与Oracle最佳动作的分列统计亦保留。

B_selected_sources.csv列出E15最高分动作、Oracle最高分动作的四项及I/B/D，并加入execute标记。原B_interaction_summary中的E15分类针对最高分候选，含被no-op门控拒绝者：例如初始Full-V负收益最高分动作48例，而真正执行错误为E17的46例，口径不能混淆。

质量误差影响这里的VV/VZ/ZV/ZZ所构成的总I以及自身能量B，和交换后混合分母变化D是不同概念。**D在某组错误中占比小，绝不意味着页面质量误差不重要。**

## D：现有Full-V到底观察到了什么

保持原在线16 probes，先逐Query计算余弦再汇总。页面代理为h，集合代理为−H，候选代理为h_r−h_a；目标分别是总残差、Value响应分量、质量分量。

| 初始Risk范围 | 总残差余弦 | Value分量余弦 | 质量分量余弦 |
|---|---:|---:|---:|
{dt}

初始集合总余弦0.409不同于E15全部反事实状态平均0.354，原因是统计范围不同，不是代理改变。后续状态集合代理与V/Z分量余弦约−0.233/+0.607。

这个结果解释了“方向有信号却选错”的一种机制：代理贴近质量分量，但没有可靠表示与它抵消的Value响应分量。高eZ余弦一部分来自共享的Ohat中心化方向结构；它**不证明能够在线预测真实d、分量幅度或最终收益**。不能据此直接部署质量分量预测器。

## C：严格区分数值单位、替换诊断与收益上限

### 统一单位与冻结方式

保留E17训练得到的λ，不再调比值或门槛。代理使用在线可得的全摘要输出能量 Chat=4Σ16||Ohat_all||²，将Iproxy、λBproxy转换为NMSE形式；再只在训练36例上拟合一个共同非负系数c同时匹配真实I、B。每例等权，例内状态动作等权；用训练例的mean(I²+B²)对拟合损失加权。

预测项为 c·64Iproxy/Chat 与 c·64λBproxy/Chat。只用同一正系数和同一在线归一化，**原E17排序与score>0门控完全不变**，审计已逐例验证；评价集没有使用真实标签或Dense能量来校准代理。真实I/B均采用原Q64 NMSE单位，不将E17的事件标签标准差误当成可在线获得的换算系数。

| 表示 | 固定λ | 训练公共c | 训练归一化联合MSE（零预测=1） |
|---|---:|---:|---:|
{calibration}

必须注意：联合校准误差接近零预测基线（评价Full-V约0.9814、均值约0.9834）。**单位统一并没有使保留的代理变准确。** 因此单项教师替换衡量的是“在这个失准代理仍保留时”的效果，不能解读为该信息本身的无条件能力上限。

### 初始Risk的受控替换

所有方案使用同一共同合法动作面板，分数>0才执行，最终均按完整64 Query真实变化分母评价。

| Full-V干预 | 平均逐例相对改善 | 退化率 | 汇总绝对收益/完整Oracle收益 |
|---|---:|---:|---:|
{ct}

含义：proxy_calibrated为原E17动作；teacherI仅替换I，teacherB仅替换B；teacherIB同时替换I/B但不给D；full_oracle使用真实I−B+D。完整Oracle也只是本面板单动作上限，不是全局最优精算集合。

**真实I＋代理B的极端失败必须保留。** 中位数改善为+3.751%，P10为−20.618%，但最差一例相对退化约1,918,962%，导致均值约−27,027.8%。该例原NMSE仅2.5971e−5，所选动作真实收益−0.498378；I=0.000859、B=0.133707、D=−0.365531。真实交互有利远不足以抵消真实能量和分母损失。独立原始残差重算通过，不是数值溢出；未裁剪、未删除。

Mean-only的单项教师干预在此轮选到同样动作，详见CSV。不能据此声称“真实I没有价值”：保留的能量估计校准很差，并且替换改变了所选动作分布。也不能将这类混合教师评分称作最优上界。

同时知道真实I/B时，初始Risk平均相对改善 **4.889%**，恢复汇总绝对Oracle收益 **84.03%**；完整Oracle平均改善5.437%。匹配16 probes的真实I/B乘4作为Q64求和估计，恢复57.19%、平均改善3.575%。所以一旦分子信息变准确，Query覆盖仍可能成为限制；这不否定E17对原代理符号失败的16/64对照结论。

### 后续集合与分母尾部

不能把初始Risk的84.03%外推到所有反事实状态。真实I/B但忽略D的按状态结果：

| 状态 | 事件数 | 平均相对改善 | 中位数改善 | 退化数 |
|---|---:|---:|---:|---:|
{st}

uniform_perturb2中最差实例是horse_field_s303_c2_l29_s2_h2_t66：原NMSE0.00042947，I−B约+0.00012182，但D约−0.36981263，最终收益−0.36969082，退化约86,080.9%。这一例使该组均值转负；全后续集合平均改善为−195.51%，全状态为−166.87%。所有值均保存。

这说明分母变化的作用依赖当前集合和新评分选中的动作。E17原评分错误中只有少数“分母翻负”，不是忽略D的普遍安全保证。大多数局部交换轨迹状态的均值仍为正，但不能据此省略人工扰动尾部。

## 研究判断与未完成范围

E18严格回答了来源问题：两类残差都存在且抵消很强；Full-V代理明显偏向质量方向，缺少页内有符号响应失真的有效信息。与其继续把完整V输出方向当成完整残差，更值得先诊断摘要未保留的Query-dependent响应。

这与K–V交叉响应方向相容，但**尚未证明低秩交叉协方差近似有效或必要**。本轮未计算原始KV协方差、未验证小logit近似、未新增缓存；若未来研究，只应另立少量已有快照诊断。也没有新Router、训练摘要、在线延迟或闭环视频结论。

当前数据来自已参与研究的场景，只有36个独立训练head–tile；共同面板及人工状态限制了外推。能量分解、教师干预和在线可预测性是不同层次，不能相互替代。

## 审计与文件

审计全部通过：144例、994集合；734个原始文件与288个E13文件身份保持；摘要独立重建、四项交互、集合能量、原始收益、教师字段隔离、训练来源、冻结校准和E17基线动作一致性均检查。最大集合能量误差 {audit['max_set_energy_error']:.3e}，独立动作收益重算误差 {audit['max_label_error']:.3e}。

- A_energy_*.csv：页面/集合分量、负交叉项、逐例及汇总。
- B_interaction_*.csv、B_selected_sources.csv：四项交互来源、错误与正确动作。
- C_performance_summary.csv、C_per_event.csv、C_selected_actions.csv、C_by_state.csv、C_worst_actions.csv：全部教师替换与尾部。
- D_directions_*.csv：在线方向与三类真实残差的关系。
- calibration_lock.json、audit.json、protocol.json：冻结校准、数值审计、定义。
- events/*/terms.npz：全部候选的分解与干预输入，完整目录保留，精简包不重复装载原始贡献及大缓存。

在原仓库环境与冻结数据可用时设置OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1，依次运行：

```sh
python -m local_experiments.e18_residual_sources.prepare
python -m local_experiments.e18_residual_sources.fit
python -m local_experiments.e18_residual_sources.evaluate
python -m local_experiments.e18_residual_sources.audit
python -m local_experiments.e18_residual_sources.report
```

prepare使用16进程，已有结果可复用。改变实验定义应另立输出版本目录；精简包源代码依赖仓库既有模块和冻结数据，并非独立数据集。E17/E18目前是本地交付，不声称GitHub默认分支已包含源码。
'''
 (OUT/'REPORT.md').write_text(text)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4));xx=np.arange(3)
 for offset,level,scope in [(-.2,'set','risk'),(.2,'candidate','risk')]:
  vals=[next(r['cosine'] for r in D if r['level']==level and r['scope']==scope and r['component']==c) for c in ['total','V','Z']];ax[0].bar(xx+offset,vals,.4,label=level)
 ax[0].set_xticks(xx,['Total residual','V-response component','Mass component']);ax[0].axhline(0,color='black',lw=.5);ax[0].set(ylabel='Signed cosine',title='Full-V observable directions: initial Risk');ax[0].legend()
 ax[1].bar(['V','Z','Cross','Total'],[.0701259358595045,.052529259818506674,-.11237973243514571,.01027546324286548]);ax[1].axhline(0,color='black',lw=.5);ax[1].set(ylabel='Mean NMSE units',title='Set residual source cancellation');fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)
 pkg=OUT/'review';pkg.mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and p.suffix in ['.csv','.json','.md','.png']:shutil.copy2(p,pkg/p.name)
 src=pkg/'source/local_experiments/e18_residual_sources';src.mkdir(parents=True,exist_ok=True)
 for p in (ROOT/'local_experiments/e18_residual_sources').glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,src/p.name)
 dump(pkg/'source_identities.json',{p.parent.name:json.loads(p.read_text())['audit'] for p in (OUT/'events').glob('*/result.json')})
 dump(pkg/'checksums.json',{str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file() and p.name!='checksums.json'})
 target=OUT.parent/'e18_residual_sources_review.tar.gz'
 with tarfile.open(target,'w:gz') as t:t.add(pkg,arcname='e18_residual_sources_review')
 print(dict(report=str(OUT/'REPORT.md'),package=str(target),bytes=target.stat().st_size))
if __name__=='__main__':run()
