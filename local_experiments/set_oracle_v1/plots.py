import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT
rows=json.loads((ROOT/'analyses/per_instance.json').read_text());fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for case,color in [('city_walk','#4388bb'),('horse_field','#df9430')]:
 sub=[r for r in rows if r['case']==case];ax[0,0].hist([100*r['gain_vs_stronger'] for r in sub],bins=np.linspace(0,100,21),histtype='step',linewidth=2,label=case,color=color)
ax[0,0].set(xlabel='Risk + exchange improvement vs stronger Mass/Risk (%)',ylabel='Head-tile instances');ax[0,0].legend()
ax[0,1].boxplot([[100*r['gain_vs_stronger'] for r in rows if r['layer']==l] for l in (1,15,29)],tick_labels=['Layer1','Layer15','Layer29'],showfliers=True);ax[0,1].set_ylabel('Improvement vs stronger baseline (%)')
for case,color in [('city_walk','#4388bb'),('horse_field','#df9430')]:
 curves=[]
 for r in rows:
  if r['case']!=case or r['first4'] is None:continue
  s=json.loads((ROOT/'searches/v1'/r['event_id']/'result.json').read_text());t=s['trajectories']['risk'];total=t[0]['nmse']-t[-1]['nmse'];curves.append([(t[0]['nmse']-t[min(k,len(t)-1)]['nmse'])/total for k in range(21)])
 a=np.array(curves);x=np.arange(21);ax[1,0].plot(x,a.mean(0),label=case,color=color);ax[1,0].fill_between(x,np.quantile(a,.1,axis=0),np.quantile(a,.9,axis=0),alpha=.15,color=color)
ax[1,0].set(xlabel='Accepted swaps (stopped trajectories held at final)',ylabel='Fraction of eventual LOCAL search improvement');ax[1,0].legend();ax[1,0].axvline(4,color='gray',ls=':')
with (ROOT/'analyses/cancellation/per_instance.csv').open() as f:ds=list(csv.DictReader(f))
x=np.array([float(d['delta_A_at_risk_denominator'])/float(d['risk_nmse']) for d in ds]);y=np.array([float(d['delta_C_at_risk_denominator'])/float(d['risk_nmse']) for d in ds]);ax[1,1].scatter(x,y,s=15,alpha=.6);ax[1,1].axhline(0,color='gray',lw=.8);ax[1,1].axvline(0,color='gray',lw=.8);ax[1,1].set(xlabel='Delta independent energy at risk Z / risk NMSE',ylabel='Delta cross-page interaction at risk Z / risk NMSE')
fig.suptitle('set_oracle_v1: 144 head-tiles, 4 trajectories; Q64 / Page64 / fixed10%');fig.savefig(ROOT/'analyses/overview.png',dpi=170);fig.savefig(ROOT/'analyses/overview.pdf');plt.close(fig)
# A compact set of per-instance exchange curves, preregistered certificate samples.
manifest=json.loads((ROOT/'manifest.json').read_text());fig,axes=plt.subplots(4,3,figsize=(13,12),layout='constrained')
for ax,eid in zip(axes.flat,manifest['global_preregistered']):
 s=json.loads((ROOT/'searches/v1'/eid/'result.json').read_text());initial=s['values']['risk']
 for name,color in [('mass','#4388bb'),('risk','#df9430'),('greedy','#37a176')]:
  t=s['trajectories'][name];ax.plot([v['step'] for v in t],[v['nmse']/initial for v in t],'-o',ms=2,label=name,color=color)
 ax.set_title(eid,fontsize=7);ax.set_xlabel('Accepted swaps');ax.set_ylabel('NMSE / initial risk NMSE');ax.legend(fontsize=6);ax.grid(alpha=.2)
fig.savefig(ROOT/'analyses/preregistered_exchange_curves.png',dpi=150);plt.close(fig)
