import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .setup import ROOT

def main():
 s=json.loads((ROOT/'summary.json').read_text());m=s['measurements'];names=['original_Dense','original_Tile20_after','optimized_Dense','optimized_Tile20_after']
 fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 ax=axes[0];gen=[m[n]['generation_seconds'] for n in names];dec=[m[n]['decode_and_transfer_seconds'] for n in names]
 ax.bar(range(4),gen,label='Generation',color=['#8c95a5','#687cad','#65a8a2','#267c76']);ax.bar(range(4),dec,bottom=gen,label='VAE + transfer (separate timing)',color='#d9dce1')
 for i,(a,b) in enumerate(zip(gen,dec)):ax.text(i,a/2,f'{a:.3f}s',ha='center',color='white',weight='bold');ax.text(i,a+b+.06,f'{a+b:.3f}s',ha='center')
 ax.set_xticks(range(4),['Old\nDense','Old\nTile20','Optimized\nDense','Optimized\nTile20']);ax.set_ylabel('Seconds per 45-frame video');ax.set_ylim(0,7.5);ax.legend(fontsize=8);ax.set_title('8 paired cases; all output latents bit-identical')
 names_en=['GEMM / FFN + projections','Norm / pointwise / other','Sparse attention','Routing scores + selection','Dense / cross attention','Q/K grouping + reorder','KV / prototypes / quantization','Merge + restore','Fused RoPE']
 groups=sorted(s['profile_group_kernel_ms'].items(),key=lambda x:-x[1]);ax=axes[1];ax.barh(names_en,[x[1] for x in groups],color='#267c76');ax.invert_yaxis();ax.set_xlabel('GPU kernel time (ms)');ax.set_title('Optimized final denoising step: all 30 layers')
 fig.savefig(ROOT/'performance.png',dpi=160);fig.savefig(ROOT/'performance.pdf');plt.close(fig)
if __name__=='__main__':main()
