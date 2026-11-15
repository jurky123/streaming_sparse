import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
root=Path('/tmp/e11_set_search');out=Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_set_search');out.mkdir(parents=True,exist_ok=True)
methods=[('mass_exchange','Mass start','#4388bb'),('risk_exchange','Risk start','#df9430'),('greedy_exchange','Greedy start','#37a176')]
for kind in ('nmse','step_gain','construction'):
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 for row,case in enumerate(('horse_field','traffic_crossing')):
  for col,layer in enumerate((1,15,29)):
   d=json.loads((root/case/f'l{layer}/search_result.json').read_text());ax=axes[row,col]
   if kind=='construction':
    t=d['greedy']['trace'];ax.plot([v['count'] for v in t],[v['nmse'] for v in t],'-o',ms=2,color='#37a176');ax.set_yscale('log');ax.set_xlabel('Exact pages (own included)');ax.set_ylabel('True NMSE');ax.axhline(d['values']['risk'],color='#df9430',ls='--',label='Risk at 41 pages');ax.axhline(d['values']['mass'],color='#4388bb',ls=':',label='Mass at 41 pages')
   else:
    for method,label,color in methods:
     t=d['exchange_curves'][method];xx=[v['step'] for v in t];yy=[v['nmse'] if kind=='nmse' else v['delta']/t[0]['nmse']*100 for v in t]
     if kind=='step_gain':xx=xx[1:];yy=yy[1:]
     ax.plot(xx,yy,'-o',ms=3,color=color,label=label,alpha=.85)
     if kind=='nmse':ax.scatter(t[-1]['step'],t[-1]['nmse'],marker='x',s=60,color=color)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.set_xlabel('Accepted swaps (0 = initial set)');ax.set_ylabel('True NMSE' if kind=='nmse' else 'Step reduction / initial NMSE (%)')
    if kind=='nmse':ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
   ax.set_title(f'{case} / layer {layer}');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle('Q64 / Page64 / 410 candidates / exact 41; seed101, chunk5, step0');filename={'nmse':'exchange_nmse','step_gain':'exchange_step_reduction','construction':'greedy_construction'}[kind];fig.savefig(out/(filename+'.png'),dpi=160);fig.savefig(out/(filename+'.pdf'));plt.close(fig)
