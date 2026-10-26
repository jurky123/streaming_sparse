import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .setup import ROOT,PARENT

def main():
 s=json.loads((ROOT/'summary.json').read_text());m=s['measurements'];names=['Opt21_Dense','Opt21_Tile20_after','Opt22_Dense','Opt22_Tile20_after']
 fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');ax=axes[0]
 gen=[m[n]['generation_seconds'] for n in names];dec=[m[n]['decode_and_transfer_seconds'] for n in names]
 ax.bar(range(4),gen,color=['#8c95a5','#687cad','#65a8a2','#267c76'],label='Generation');ax.bar(range(4),dec,bottom=gen,color='#d9dce1',label='VAE + transfer, measured separately')
 for i,(a,b) in enumerate(zip(gen,dec)):ax.text(i,a/2,f'{a:.3f}s',ha='center',color='white',weight='bold');ax.text(i,a+b+.05,f'{a+b:.3f}s',ha='center')
 ax.set_xticks(range(4),['Opt21\nDense','Opt21\nTile20','Opt22\nDense','Opt22\nTile20']);ax.set_ylim(0,6.3);ax.set_ylabel('Seconds per 45-frame video');ax.legend(fontsize=8);ax.set_title('8 paired cases; all output latents bit-identical')
 old=json.loads((PARENT/'final/case0/profile.json').read_text());new=json.loads((ROOT/'final/case0/profile.json').read_text());groups=s['profile_group_kernel_ms'];top=sorted(groups.items(),key=lambda p:-p[1])
 mapping={'GEMM：FFN及线性投影':'GEMM: FFN + projections','稀疏attention主循环':'Sparse attention','路由评分与选择':'Routing scores + selection','其他归一化、逐元素及管理':'Other norms / pointwise / management','Dense/交叉attention':'Dense / cross attention','Q/K分组与重排':'Q/K grouping + reorder','KV/原型与量化':'KV / prototypes / quantization','融合调制与残差':'Fused modulation + residual','融合RMSNorm（原归约顺序）':'Ordered fused RMSNorm','融合分段合并/还原':'Merge + restore','融合RoPE':'Fused RoPE'}
 ax=axes[1];ax.barh([mapping[k] for k,v in top],[v for k,v in top],color='#267c76');ax.invert_yaxis();ax.set_xlabel('GPU kernel time (ms)');ax.set_title(f"Final denoising step: {old['kernel_sum_ms']:.1f} -> {new['kernel_sum_ms']:.1f} ms")
 fig.savefig(ROOT/'performance.png',dpi=160);fig.savefig(ROOT/'performance.pdf');plt.close(fig)
if __name__=='__main__':main()
