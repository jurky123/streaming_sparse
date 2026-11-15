"""Scientific static comparison plot; independent datasets and bounds stay visible."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/shuchenweng/jzk/streaming_sparse/artifacts/e11_compact');rows=json.loads((root/'comparison.json').read_text());metrics=json.loads((root/'matched_metrics.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=np.arange(3);w=.24
for off,key,label,color in [(-1,'mass_nmse','Mass','#6699cc'),(0,'risk_nmse','Risk','#e5ae38'),(1,'incumbent_UB','Best feasible','#46a078')]:
 ax[0].bar(x+off*w,[r[key]/r['risk_nmse'] for r in rows],w,label=label,color=color)
for i,r in enumerate(rows):
 if r['global_LB']>0:ax[0].plot([i+w,i+w],[r['global_LB']/r['risk_nmse'],r['incumbent_UB']/r['risk_nmse']],color='black',lw=3);ax[0].scatter(i+w,r['global_LB']/r['risk_nmse'],marker='_',color='black')
ax[0].axhline(1,color='gray',lw=.6);ax[0].set_xticks(x,[f"Layer {r['layer']}" for r in rows]);ax[0].set_ylabel('True NMSE / risk NMSE');ax[0].set_title('Q64 / Page64; 410 pages; exact 41');ax[0].legend(fontsize=8)
for i,layer in enumerate((1,15,29)):
 for j,method in enumerate(('risk','best_feasible')):
  r=next(m for m in metrics if m['layer']==layer and m['method']==method);base=next(m for m in metrics if m['layer']==layer and m['method']=='risk')['nmse'];pos=i+(j-.5)*.3
  ax[1].bar(pos,r['A_nmse']/base,.28,color='#826ab3',label='Independent A / Z²' if i==0 and j==0 else None);ax[1].bar(pos,r['C_nmse']/base,.28,bottom=r['A_nmse']/base,color='#66aa99',label='Cross term C / Z²' if i==0 and j==0 else None)
ax[1].set_xticks(x,['L1\nRisk | Best','L15\nRisk | Best','L29\nRisk | Best']);ax[1].set_title('Less positive cross-page accumulation');ax[1].set_ylabel('Contribution / risk NMSE');ax[1].legend(fontsize=8);fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)
