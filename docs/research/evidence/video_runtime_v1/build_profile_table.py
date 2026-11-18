"""Rebuild the published module tables from the repository root."""
from pathlib import Path
from collections import Counter
import json,csv
root=Path('artifacts/video_runtime_v1')
out=Path('docs/research/evidence/video_runtime_v1')
src=root/'compensation_scale_v1'
base,dense=(json.loads((src/(k+'_timeline.json')).read_text()) for k in ('base','dense'))
budget=json.loads((src/'attention_budget.json').read_text())
depl=json.loads((root/'anchor_accum_v1/deployment.json').read_text())
theory=json.loads((root/'formal_short_15/theoretical_cost.json').read_text())
def group(k):
 if 'pytorch_flash::flash_fwd_kernel' in k:return '保留的原生自注意力'
 if k=='_tail':return '稀疏精算与补偿融合'
 if k.startswith('_ZN7cutlass6Kernel'):return 'FFN 第一层投影与 GELU'
 if k.startswith('ampere_bf16_s16816gemm'):return '其余 BF16 GEMM（投影与 FFN 输出等）'
 if k.startswith('repeated('):return '重复文本交叉注意力'
 if k.startswith('void rms_row_kernel<true>'):return '自注意力 Q/K RMS 与 RoPE'
 if k.startswith('void rms_row_kernel<false>'):return 'RMS（无 RoPE）'
 if k.startswith('void normalize_modulate'):return 'LayerNorm 与调制'
 if k in ('_partial','_finish'):return 'Risk 分块打分与归一化'
 if k in ('_page_summary','_rep_stats') or 'pool_rep(' in k:return '页摘要与代表统计'
 if k=='_sort_rows' or 'sort_pages<' in k:return '排序与选页'
 if k.startswith(('void direction_kernel','void chain<','void staged_columns','_features')):return '分组特征、方向与投影'
 if k.startswith('ampere_sgemm'):return 'FP32 GEMM'
 if k.startswith(('Memcpy','Memset')):return '显式复制与清零'
 if 'at::native' in k:return 'PyTorch 通用逐元素与归约'
 return '其余设备核'
def total(rows):
 c=Counter()
 for a in rows:c.update(a['by_name_ms'])
 return c
b,d=total(base),total(dense)
g=Counter()
for k,v in b.items():g[group(k)]+=v
q,w,f=4608,1536,8960
layers=1350
peak=312e9 # FLOPs per ms
pcount=[min(c+1,6)*72 for c in range(9)]
flash=sum(4*12*q*p*64*128*(150 if c<2 else 34) for c,p in enumerate(pcount))
models={
'稀疏精算与补偿融合':(budget['total']['main_fused_flops']/peak,'按页填充，双项 AV，109.93 TFLOP 主项'),
'保留的原生自注意力':(flash/peak,'前两段每段 150 次，后七段各 34 次'),
'FFN 第一层投影与 GELU':(layers*2*q*w*f/peak,'第一层 GEMM 主项，忽略 GELU'),
'其余 BF16 GEMM（投影与 FFN 输出等）':(layers*(2*q*w*f+12*q*w*w)/peak,'FFN 输出及 self Q/K/V/O、cross Q/O 六个投影主项'),
'重复文本交叉注意力':(layers*4*q*64*w/peak,'每次处理前 64 个文本位置的 QK 和 AV'),
'自注意力 Q/K RMS 与 RoPE':(layers*2*q*w*4/1.555e9,'按两个 BF16 张量各读写一遍的带宽估算'),
'LayerNorm 与调制':(layers*q*w*(4+8+8)/1.555e9,'三个阶段每元素合计 20 B 主张量逻辑读写'),
'RMS（无 RoPE）':(layers*q*w*4/1.555e9,'视频查询 RMS 每元素读写 4 B；忽略文本 K')}
rows=[]
def add(name,v,t,n):
 rows.append((name,v,t,None if t is None else v-t,None if not t else 100*(v/t-1),n))
for k,v in g.most_common():
 t,n=models.get(k,(None,'缺少逐调用形状或字节/指令计数'));add(k,v,t,n)
def fmt(x):return '—' if x is None else f'{x:.2f}'
head=['模块','实测 ms','理论参考 ms','绝对差 ms','相对差 %','理论口径']
def table(rr):
 return '| '+' | '.join(head)+' |\n|---|---:|---:|---:|---:|---|\n'+''.join('| '+' | '.join([r[0],*map(fmt,r[1:5]),r[-1]])+' |\n' for r in rr)
modrows=rows.copy();rows.clear()
ref=depl['dense']['median']*1000;obs=depl['base']['median']*1000
smax=max(theory['generation_dominant_flop_speedup']['9'])
add('base 对同卡 Dense 的理论主项速度',obs,ref/smax,f'Dense {ref:.2f} ms ÷ {smax:.5f}×')
add('base 对项目验收线',obs,depl['target_seconds']*1000,'Dense ÷ 1.40034×')
add('融合注意力对同形状 Dense 吞吐',g['稀疏精算与补偿融合'],budget['total']['at_dense_rate_ms'],'逐段用被替换的 Dense 时间按 FLOP 比例折算')
refrows=rows.copy()
text='# 105 帧 base 模块 profile 与理论耗时差距\n\ncity_walk / seed 701，A100-SXM4-40GB，GPU7。base 已启用首页缩放和精算分母列累加。正式生成预热后 5 次中位数为 5053.12 ms，相对同卡原生 Dense 为 1.27983×；计时截止于最终 latent。\n\n## 整条视频与实际 Dense 吞吐参照\n\n'+table(refrows)+'\n绝对差为实测减参考，相对差为绝对差除以参考。融合核这行是设备累计时间；前两行是整条视频墙钟时间。\n\n## 分模块 GPU 累计时间\n\n'+table(modrows)+f'\n全部设备事件累计 {sum(g.values()):.2f} ms，按 kernel 名互斥分组；并行流有重叠。每段设备区间的并集累计 {sum(x["active_ms"] for x in base):.2f} ms。设备区间与正式墙钟计时来自独立运行。\n\n矩阵乘按 BF16、FMA=2 和 312 TFLOP/s 计算；带宽估算按 1555 GB/s，假设主张量流经 HBM，缓存命中会改变结果。每层前向取 4608 token、宽 1536、FFN 宽 8960，共 9×5×30 次层前向。主项模型忽略非线性、缓存初始化及调度开销。表中空值表示缺少可信理论分母。Risk 满窗口已有独立计数，见下表。\n\n'
roof=root/'risk_remainder_v1/roof.csv'
import io
rr=list(csv.DictReader(io.StringIO('"ID"'+roof.read_text().split('"ID"',1)[1])))
v={r['Metric Name']:float(r['Metric Value'].replace(',','')) for r in rr}
ms=v['gpu__time_duration.sum']/1e6
tb=(v['dram__bytes_read.sum']+v['dram__bytes_write.sum'])/1.555e9
tc=2*12*72*16*432*4*128/peak
rows.clear();add('Risk 分块核，432 页单次采样',ms,max(tb,tc),'取 QK 有效主项 / 312 TFLOP/s 与实测 HBM 字节 / 1555 GB/s 的较大值；忽略标量计算')
text+='## 单次硬件计数\n\n'+table(rows)+'\n此行来自同形状 Risk 核的独立 NCU 运行。整条视频还包含其他页数及归一化，不能按满窗口结果直接外推。\n\n[模块表 CSV](PROFILE_MODULES_105.csv) · [逐核明细 CSV](PROFILE_KERNELS_105.csv) · [计算口径与逐段数据](../../../../artifacts/video_runtime_v1/compensation_scale_v1/attention_budget.json) · [部署计时](../../../../artifacts/video_runtime_v1/anchor_accum_v1/deployment.json) · [NVIDIA A100 规格](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)\n'
(out/'PROFILE_MODULES_105.zh.md').write_text(text)
with (out/'PROFILE_KERNELS_105.csv').open('w') as h:
 wr=csv.writer(h);wr.writerow(['模块','CUDA kernel / event','base_ms','dense_ms'])
 for k in sorted(b.keys()|d.keys(),key=lambda k:-b[k]):wr.writerow([group(k),k,f'{b[k]:.6f}',f'{d[k]:.6f}'])

with (out/'PROFILE_MODULES_105.csv').open('w') as h:
 wr=csv.writer(h);wr.writerow(['范围',*head])
 for label,rr in [('整段及 Dense 参照',refrows),('设备时间线',modrows),('独立 Risk 硬件采样',rows)]:
  for name,observed,ideal,gap,pct,note in rr:
   wr.writerow([label,name,fmt(observed),fmt(ideal),fmt(gap),fmt(pct),note])
