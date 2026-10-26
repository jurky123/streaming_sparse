import json
from pathlib import Path
def plot(folder):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=json.loads((folder/'curve.json').read_text());logs=[json.loads(s)for s in(folder/'train.jsonl').read_text().splitlines()if s.strip()]
 fig,ax=plt.subplots(figsize=(8,4));window=25
 ax.plot([x['updates']for x in rows],[x['nmse']for x in rows],'o-',label='Val: actual native attention NMSE')
 smooth=[logs[i:i+window]for i in range(0,len(logs),window)]
 ax.plot([x[-1]['updates']for x in smooth],[sum(r['actual_nmse']for r in x)/len(x)for x in smooth],alpha=.6,label='Train: mean25 updates')
 ax.set_xlabel('Updates');ax.set_ylabel('Actual attention NMSE');ax.set_title(str(folder.parent.name)+' / '+folder.name);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(folder/'loss_curve.png');plt.close(fig)
