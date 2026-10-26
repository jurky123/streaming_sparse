"""Actual train/validation loss curves; called only at validation boundaries."""
import json
from pathlib import Path

def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);val=json.loads((folder/'curve.json').read_text())
    train=[json.loads(s)for s in (folder/'train.jsonl').read_text().splitlines()if s.strip()]
    if not train:return
    # Recovery can append repeated steps; show the most recent completed value.
    train=list({r['updates']:r for r in train}.values());train.sort(key=lambda r:r['updates'])
    x=[r['updates']for r in train];y=[r['actual_nmse']for r in train]
    smooth=[sum(y[max(0,i-24):i+1])/len(y[max(0,i-24):i+1])for i in range(len(y))]
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(x,y,alpha=.2,label='Train batch (actual kernel)');axes[0].plot(x,smooth,label='Train rolling 25')
    axes[1].plot([r['updates']for r in val],[r['nmse']for r in val],'o-',label='Val 96 events')
    axes[1].axhline(val[0]['nmse'],ls='--',color='gray',label='Initial')
    for ax in axes:ax.set_xlabel('Update');ax.set_ylabel('Actual attention NMSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(folder.parent.name+' / '+folder.name);fig.tight_layout();fig.savefig(folder/'loss_curve.png',dpi=140);plt.close(fig)

def main():
    from .setup import ROOT
    for f in (ROOT/'training').glob('layer*/*/curve.json'):
        if (f.parent/'train.jsonl').exists():plot(f.parent)

if __name__=='__main__':main()
