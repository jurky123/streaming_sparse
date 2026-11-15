from .report import *
def main():
 rows=[];su=[]
 for seed in [17,29]:
  dest=DEST/f's{seed}_uniform';assert (dest/'complete.json').exists()
  rr=json.loads((dest/'decisions.json').read_text());rows.extend(r for r in rr if r['method'].startswith('Local'))
  for method in ['Local-Oracle64','Local-Shared','Local-LOTO']:
   for policy in ['Set','Independent']:su.append(dict(seed=seed,method=method,policy=policy,**aggregate([r for r in rr if r['method']==method and r['policy']==policy])))
 table('uniform_local_per_event.csv',rows);table('uniform_local_summary.csv',su)
 with (DEST/'REPORT.md').open('a') as f:
  f.write('\n\n## 等权局部拟合敏感性（不参与选择）\n\n主比较两种目标都沿用mass加权LS。下表只移除局部目标的mass权重，其余设置相同。此附加检查在读取主结果前登记，用于区分目标局部性与全局回归权重；不把它与主比较混合选择最优结果。\n\n|局部等权Set方法|seed17 Oracle恢复|seed29 Oracle恢复|\n|---|---:|---:|\n')
  for method in ['Local-Oracle64','Local-Shared','Local-LOTO']:
   x=[next(r for r in su if r['method']==method and r['seed']==s and r['policy']=='Set') for s in [17,29]];f.write(f"|{method}|{x[0]['R_executed']:.2%}|{x[1]['R_executed']:.2%}|\n")
  f.write('\n不能认为移除全局权重必然更好，等权结果随种子变化明显。主结果说明在现有质量加权目标下，解析中心修正可以保留更多跨Query决策信息，不说明任意局部回归都更稳定。\n')
 # Common-objective scientific plot, not an interactive UI.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 base=list(csv.DictReader((DEST/'decision_summary.csv').open()));names=['Centered-Oracle64','Local-Oracle64','Centered-Shared','Local-Shared','Centered-LOTO','Local-LOTO','P1'];fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
 for ax,seed in zip(axs,[17,29]):
  vals=[100*float(next(r['R_executed'] for r in base if int(r['seed'])==seed and r['method']==n and r['policy']=='Set')) for n in names];ax.bar(range(len(names)),vals,color=['#8c99aa','#218b82','#8c99aa','#218b82','#8c99aa','#218b82','#dfac42']);ax.set_xticks(range(len(names)),['Global\nEvent','Local\nEvent','Global\nShared','Local\nShared','Global\nHoldout','Local\nHoldout','True16\nP1'],rotation=30);ax.set_title(f'Frozen C seed {seed}');ax.axhline(0,color='black',lw=.7);ax.set_ylim(0,75)
  for i,v in enumerate(vals):ax.text(i,v+.7,f'{v:.1f}',ha='center',fontsize=8)
 axs[0].set_ylabel('True Q64 Oracle gain recovery (%)');fig.suptitle('Offline teacher diagnostic: true beta; M16 decisions; fixed Medium pool');fig.tight_layout();fig.savefig(DEST/'capability.png',dpi=180);plt.close(fig)
 with (DEST/'REPORT.md').open('a') as f:
  f.write('\n\n## 结论边界\n\n页面局部目标的主比较保留了更多共享及留出tile决策收益，支持全局中心化目标与页面固定系数之间存在结构失配这一机制假设。但这里只观察到教师系数的可复用性，没有训练或证明从R4到系数的在线可预测性。Local-Shared不是无标签泛化；Local-LOTO仍使用真实beta、其他上下文教师以及当前页逐事件教师拟合。\n\n新构造同时把全局摘要中心和真实质量比放入解析路径，不再强迫五项因子化一并拟合它们。因此改善不能纯粹归因为“去掉全局中心”这单一因素。局部deltaO自身的跨Query拟合NMSE仍可能大于1，系数形状变化也没有消失；主要成功是决策保留，不是高保真响应重建。\n\n本轮不自动启动Page Encoder训练。若随后训练，必须同时测试真实/预测beta，保持旧摘要和当前page处理不变，并证明Set同时优于no-op与同信息Independent。\n')
 print(json.dumps(su,indent=2))
if __name__=='__main__':main()
