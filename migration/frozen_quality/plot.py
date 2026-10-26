"""Standalone figures from the complete measured quality assessment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args();root=Path(args.directory)
    summary=json.loads((root/'summary.json').read_text());assert summary['passed']
    names=['mass','K200','V200','Kbest','Kpaused'];colors=['#888888','#2864b7','#df7b25','#319468','#8c62ad']
    labels=['Mass / V-best@0','K warm@200','V warm@200','K main-best','K paused@1039']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0];values=[summary['offline'][n]['prompt_equal_error']for n in names]
    ax.bar(np.arange(5),values,color=colors);ax.set_xticks(np.arange(5),labels,rotation=20,ha='right');ax.set_ylabel('Prompt-equal x0 NMSE (lower is better)');ax.set_title('Fixed external replay: 192 inputs')
    ax.set_ylim(0,max(values)*1.15)
    for i,v in enumerate(values):ax.text(i,v,f'{v:.7f}',ha='center',va='bottom',fontsize=8)
    ax=axes[0,1]
    for n,c,l in zip(names,colors,labels):ax.plot([1,2,3],[summary['offline'][n]['history_tokens'][str(h)]for h in [4680,9360,14040]],'o-',color=c,label=l)
    ax.set_xticks([1,2,3]);ax.set_xlabel('Visible history chunks');ax.set_ylabel('Prompt-equal x0 NMSE');ax.set_title('Replay error by history length');ax.grid(alpha=.2)
    for pos,key,ylabel,title in [(axes[1,0],'lpips','LPIPS vs paired Dense (lower is better)','Closed loop: error accumulation'),(axes[1,1],'psnr_db','RGB PSNR dB (higher is better)','Closed loop: decoded RGB similarity')]:
        for n,c,l in zip(names,colors,labels):
            values=[r[key]for r in summary['videos'][n]['chunks']]
            if key=='psnr_db':pos.plot([2,3,4],values[1:],'o-',color=c,label=l)
            else:pos.plot([1,2,3,4],values,'o-',color=c,label=l)
        pos.set_xticks([1,2,3,4]if key=='lpips'else[2,3,4]);pos.set_xlabel('Generated chunk');pos.set_ylabel(ylabel);pos.set_title(title);pos.grid(alpha=.2)
    axes[1,0].legend(fontsize=8)
    fig.suptitle('Frozen backbone: 8 Val prompts x 2 generation seeds | 480p | only layers 4/14/24 sparse\nK200/V200 are budget-matched; main K snapshots received additional router training',fontsize=12)
    fig.savefig(root/'quality_comparison.png',dpi=180);fig.savefig(root/'quality_comparison.pdf');plt.close(fig)


if __name__=='__main__':main()
