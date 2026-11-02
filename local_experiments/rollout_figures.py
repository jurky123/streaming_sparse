"""Static diagnostics from existing uncompressed-RGB metrics and raw timings."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .rollout_analyze import ROOT,CASES,METHODS,write,sha

def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    colors=['#000000','#666666','#A23B00','#0072B2','#006B56','#7B3294']
    styles=['-','--',':','-.','--',':']
    for case in CASES:
        metrics={m:json.loads((ROOT/case/f'{m}.quality.json').read_text()) for m in METHODS if (ROOT/case/f'{m}.rep0.json').exists()}
        if not metrics:continue
        with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
            fig,axs=plt.subplots(3,1,figsize=(11,8),layout='constrained',sharex=True)
            for j,m in enumerate(METHODS):
                if m not in metrics:continue
                f=metrics[m]['per_frame'];x=np.arange(480)/16
                for ax,key,label in zip(axs,['lpips','brightness','motion'],['LPIPS vs native dense','Mean luminance [0,1]','Frame difference L1 [0,1]']):
                    ax.plot(x,f[key],color=colors[j],linestyle=styles[j],linewidth=1,label=m)
                    ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
            axs[0].axhline(.10,color='#777777',linewidth=.8,linestyle=':',label='main mean gate (not per-frame gate)')
            axs[0].legend(ncols=3,fontsize=8);axs[-1].set_xlabel('Video time (s)');axs[-1].set_xlim(0,30)
            fig.suptitle(case+' | one seed; all frames shown; no smoothing\nLuminance and motion are diagnostics, not standalone quality scores')
            for ext in ['png','pdf']:fig.savefig(out/f'{case}.temporal.{ext}',dpi=160,facecolor='white')
            plt.close(fig)
    summary=json.loads((ROOT/'summary.json').read_text());rows=summary['rows']
    with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
        fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained',sharey=True)
        for ax,case in zip(axs.flat,CASES):
            for j,m in enumerate(METHODS):
                r=next((r for r in rows if r['case']==case and r['method']==m),None)
                if r:
                    ax.errorbar(j,r['total_s'],yerr=[[r['total_s']-r['min_s']],[r['max_s']-r['total_s']]],fmt='o',color=colors[j],capsize=4)
                    ax.annotate(f"n={r['n']}",(j,r['max_s']),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
                else:ax.text(j,.03,'pending',transform=ax.get_xaxis_transform(),rotation=90,ha='center',fontsize=8,color='#666666')
            ax.set_xlim(-.4,5.4);ax.set_title(case);ax.set_xticks(range(6),METHODS,rotation=25,ha='right');ax.set_ylim(bottom=0);ax.set_ylabel('Generation + VAE (s)');ax.grid(axis='y',alpha=.2)
        axs[0,0].set_ylim(0,1.15*max((r['max_s'] for r in rows),default=1))
        fig.suptitle('Measured median and min–max; repeated identical videos, not confidence intervals\nShared H200; incomplete repeats remain provisional')
        for ext in ['png','pdf']:fig.savefig(out/f'timing.{ext}',dpi=160,facecolor='white')
        plt.close(fig)
    write(out/'provenance.json',json.dumps(dict(script_sha256=sha(Path(__file__)),source_summary_sha256=sha(ROOT/'summary.json'),units='one prompt/seed per case; up to 3 time repetitions',transformations='No temporal smoothing. Error bars=min/max, markers=median. Missing shown as pending, not zero.',purpose='Internal research report, no journal compliance claimed.'),indent=2))

if __name__=='__main__':main()
