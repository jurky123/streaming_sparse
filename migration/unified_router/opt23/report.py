"""Finalize auditable scaling measurements and standalone figures."""
import csv,json,statistics
from pathlib import Path
from .setup import ROOT,TRAINING,PARENT,CASES,tag
from ..common import atomic_json,sha

def main():
 rows=[];allrows=[]
 for h,w,c in CASES:
  name=tag(h,w,c);rr=[json.loads((ROOT/'bench'/name/f'layer{layer:02d}.json').read_text()) for layer in [1,14,24,29]];allrows+=rr
  assert all(x['validation']['graph_exact'] and x['validation']['max_nmse']<1e-4 for x in rr)
  assert json.loads((ROOT/'capture'/name/'completed.json').read_text())['completed']
  mean=lambda field,key:statistics.mean(x[field][key] for x in rr)
  times={k:mean('median_ms',k) for k in rr[0]['median_ms']};work={k:mean('work',k) for k,v in rr[0]['work'].items() if isinstance(v,(int,float))};components={k:mean('components_ms',k) for k in rr[0]['components_ms']}
  d=times['dense_flash'];s=times['sparse_attention'];p=times['sparse_pipeline'];r=times['router']
  row=dict(case=name,height=h,width=w,chunks=c,history_chunks=c-1,query_tokens=rr[0]['query_tokens'],kv_tokens=rr[0]['kv_tokens'],median_ms=times,work=work,components_ms=components,attention_time_fraction=s/d,pipeline_time_fraction=p/d,attention_speedup=d/s,pipeline_speedup=d/p,oracle_time_fraction=times['oracle_contiguous20_flash']/d,router_time_fraction=r/d,router_logical_flop_fraction=work['router_matmul_logical_flops']/work['dense_matmul_flops'],router_padded_flop_fraction=work['router_matmul_padded_flops']/work['dense_matmul_flops'],attention_nominal_tflops=work['sparse_attention_matmul_flops']/s/1e9,dense_nominal_tflops=work['dense_matmul_flops']/d/1e9,attention_nmse_vs_dense=mean('validation','attention_nmse_vs_dense'),layer_time_ranges={k:[min(x['median_ms'][k] for x in rr),max(x['median_ms'][k] for x in rr)] for k in times})
  rows.append(row)
 probes=[json.loads(p.read_text()) for p in sorted(ROOT.glob('wide_probe*.json'))];assert len(probes)==2 and all(p['completed'] and len(p['rows'])==6 for p in probes)
 ckpts=json.loads((TRAINING/'chosen_checkpoints.json').read_text());assert all(sha(Path(p))==s for p,s in ckpts.items())
 parent=json.loads((PARENT/'FINAL_VERIFICATION.json').read_text());assert all(sha(Path(p))==s for p,s in parent['files'].items())
 assert all(x['checkpoint_sha256']==ckpts[x['checkpoint']] for x in allrows)
 summary=dict(completed=True,scope='A100 40GB, actual Dense closed-loop QKV, one fixed validation prompt seed0, final chunk step3, layers1/14/24/29. Operator timings, NOT high-resolution sparse video quality or whole-model generation timings.',timing='Forced BF16 PyTorch FlashAttention vs INT8-QK/BF16-PV sparse; CUDA graphs warm replay, 5 balanced rounds x40 repetitions, per-layer median then arithmetic mean across4layers. Current Q/K/V preparation and route included in sparse_pipeline; historical preparation excluded and separately reported. No host/file/decode/model-loading time.',historical_scope='7 chunks =21 latent frames =81 RGB frames; six historical chunks at final chunk, maximum for the evaluated native81-frame configuration, not an architectural context limit.',oracle_scope='Contiguous first20% KV FlashAttention is a performance-only control, not a valid quality-preserving routing alternative.',baseline_full_video=dict(source=str(PARENT),dense_generation_seconds=2.623494,tile20_generation_seconds=2.340766,paired_cases=8,resolution=[480,832],rgb_frames=45,unchanged_by_opt23=True),cases=rows,wide_probes=probes,hardware_counters='Nsight Compute failed with ERR_NVGPUCTRPERM; no measured HBM/L2 bandwidth or occupancy claim.',next_work=['Fuse or compress fine-route score materialization plus tile aggregation; measure extra recomputation against memory savings.','Optimize coarse/top-k reductions, not only projection GEMM.','Improve sparse attention KV reuse and staging while keeping 32-query selection unchanged; naive wider K alone failed.'])
 atomic_json(ROOT/'summary.json',summary)
 with (ROOT/'scaling.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['case','history_chunks','Q','KV','keep','executed','dense_ms','attention_ms','router_ms','pipeline_ms','attention_speedup','pipeline_speedup','oracle_ms','history_prepare_ms'])
  for x in rows:
   t=x['median_ms'];writer.writerow([x['case'],x['history_chunks'],x['query_tokens'],x['kv_tokens'],x['work']['true_keep'],x['work']['executed_fraction'],t['dense_flash'],t['sparse_attention'],t['router'],t['sparse_pipeline'],x['attention_speedup'],x['pipeline_speedup'],t['oracle_contiguous20_flash'],x['components_ms']['history_prepare']])
 lines=['Opt23：20% Tile20 的 attention / 路由兑现率与规模扩展','',
 '结论：实际 attention 执行量20.07%～20.62%，耗时28.19%～30.13%，加速3.32～3.55倍。完整稀疏模块1.70～2.11倍，仍未兑现约5倍的理想值。',
 '连续20% KV FlashAttention对照耗时20.12%～20.80%，表明本次尺寸已能接近线性缩放，当前实现仍有可观察的效率差距。',
 '对照仅表示连续KV情况下的性能；其KV选择不满足质量要求，不能用它替代当前路由。','',
 '测速口径：A100-SXM4-40GB，单提示词seed0，真实Dense闭环生成的最后chunk、step3，层1/14/24/29。',
 '每层5轮×40次CUDA graph热重放取中位数，再对4层取均值。Dense强制PyTorch FlashAttention BF16。',
 '纯稀疏attention包含INT8-QK/BF16-PV、split合并与query复原，不含路由、Q/K重排、K量化。',
 '完整稀疏模块包含当前KV分组/原型/量化、Q分组、路由、attention；历史预处理单列。',
 '这是算子实验，不是高分辨率稀疏视频质量实验，也不是整模型/视频生成加速。','',
 '分辨率        历史chunk    Q/KV tokens        Dense ms  稀疏att ms  路由ms  完整模块ms  att/Dense  模块加速']
 for x in rows:
  t=x['median_ms'];lines.append(f"{x['height']}x{x['width']:<7} {x['history_chunks']:>4}       {x['query_tokens']:>5}/{x['kv_tokens']:<6}     {t['dense_flash']:7.3f}   {t['sparse_attention']:7.3f}   {t['router']:7.3f}   {t['sparse_pipeline']:7.3f}     {100*x['attention_time_fraction']:5.2f}%    {x['pipeline_speedup']:.2f}x")
 lines+=['','3历史chunk对应现有45帧测试的最后段；6历史chunk对应81帧配置的最后段。81帧不是模型理论最大上下文。','', '路由为什么仍贵：']
 for x in [rows[0],rows[-1]]:
  lines.append(f"{x['case']}：路由矩阵乘逻辑量是Dense attention的{x['router_logical_flop_fraction']*100:.2f}%，含tensor-core填充后{x['router_padded_flop_fraction']*100:.2f}%，但耗时{x['router_time_fraction']*100:.2f}%。fine score中间张量{x['work']['route_score_bytes']/1e6:.1f} MB。")
 largest=next(x for x in allrows if x['case']==rows[-1]['case'] and x['layer']==24);prof={x['name']:x['ms'] for x in largest['kernel_profile']}
 lines+=['矩阵乘FLOPs未包含exp/log、softmax归约、top-k排序、索引和数据搬运，不能用它单独预测路由耗时。',f"960p/6历史、层24的热kernel profile：attention主kernel {prof['_shared']:.2f} ms；coarse {prof['_coarse']:.2f}；fine {prof['_fine_age']:.2f}；coarse top-k {prof['_top']:.2f}；tile归约/选择 {prof['_tiles']:.2f} ms。",'fine分数645MB，至少需要写出并再次读入；但这不是唯一瓶颈，排序与归约也占明显时间。','',
 'attention效率差距：32-query共享选择配合逐64-token页循环，收益受小tile复用、页地址计算、在线softmax更新与访存调度约束。',
 '实测逻辑attention FLOPs/耗时的有效吞吐约为Dense的68%～72%；INT8-QK与BF16-PV混合，不能把该吞吐解释成单一精度峰值利用率。',
 'Nsight Compute因服务器ERR_NVGPUCTRPERM未得到硬件计数器，因此不将“带宽已满”“occupancy不足”等推断当成硬件实证。','',
 '新增尝试：保持所有32-query选择不变，每次处理2页或4页，扫描6种宽度/warp/stage组合。']
 for p in probes:
  best=max(p['rows'],key=lambda x:x['speedup']);lines.append(f"{p['case']}最佳：{best['options']}，原版{best['baseline_ms']:.3f} ms → {best['ms']:.3f} ms（{best['speedup']:.3f}x）；NMSE={best['nmse_vs_parent']:.3g}。全部变慢，未切换生产实现。")
 lines+=['','验证：基准分辨率2种历史×4层与旧实现selection/output逐元素相等；所有24组graph/eager逐元素相等。',
 '所有24组均用独立PyTorch选页量化attention核验9个tile（涵盖首/中/末tile和3个head）；误差阈值1e-4，实际最大值见FINAL_VERIFICATION.json。',
 '480x832/4chunk采集适配器的整段Dense latent与原有Dense结果逐元素相等。冻结checkpoint未改。','',
 '下一步优先：fine-score与tile归约融合/压缩，避免大中间张量；coarse/top-k单独优化；attention改善KV复用与加载流水，而非直接放宽选择或增大K。',
 '重算式融合可能增加矩阵乘和exp/log，需实测权衡，不能仅凭少物化就声称更快。','',
 '复现：',
 './opt23_status.sh',
 'CUDA_VISIBLE_DEVICES=0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt23.capture --height 480 --width 832 --chunks 4',
 'CUDA_VISIBLE_DEVICES=0 streaming_sparse/.venv/bin/python -u -m migration.runtime.bootstrap migration.unified_router.opt23.bench --height 480 --width 832 --chunks 4 --layer 24',
 'streaming_sparse/.venv/bin/python -m migration.unified_router.opt23.report',
 '原始数据：capture/；分层5轮计时与profile：bench/；汇总：summary.json、scaling.csv；宽K失败实验：wide_probe_*.json。']
 (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 plot(rows,largest)
 files={str(p):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))};files[str(Path('opt23_status.sh').resolve())]=sha(Path('opt23_status.sh'))
 atomic_json(ROOT/'source_manifest.json',files)
 atomic_json(ROOT/'FINAL_VERIFICATION.json',dict(completed=True,captures=6,measured_blocks=len(allrows),timing_rounds_per_block=5,graph_replays_per_round=40,reference_tiles=9*len(allrows),max_reference_nmse=max(x['validation']['max_nmse']for x in allrows),base_production_exact_cases=sum(x['validation'].get('production_bit_exact',False)for x in allrows),all_graphs_exact=True,all58_checkpoints_unchanged=True,opt22_sources_unchanged=True,wide_probe_cases=2,wide_probe_configurations=12,wide_probe_not_adopted=True,new_high_resolution_video_quality_evaluations=0,files=files))
 print(ROOT/'REPORT.txt')

def plot(rows,largest):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,(ax,bx)=plt.subplots(2,1,figsize=(11,9),gridspec_kw={'height_ratios':[1.3,1]})
 x=np.arange(len(rows));att=np.array([r['attention_time_fraction']*100 for r in rows]);route=np.array([r['router_time_fraction']*100 for r in rows]);full=np.array([r['pipeline_time_fraction']*100 for r in rows]);other=full-att-route
 ax.bar(x,att,label='Sparse attention + merge',color='#2468a2');ax.bar(x,route,bottom=att,label='Router',color='#e59836');ax.bar(x,other,bottom=att+route,label='Other pipeline overhead (residual)',color='#aab5bf')
 ax.plot(x,[r['oracle_time_fraction']*100 for r in rows],'o--',color='#32936f',label='Contiguous 20% FlashAttention control')
 ax.axhline(20,color='black',ls=':',lw=1,label='20% ideal time');ax.axhline(100,color='#888888',lw=1)
 for i,r in enumerate(rows):ax.text(i,full[i]+2,f"{r['pipeline_speedup']:.2f}x",ha='center',weight='bold')
 ax.set_xticks(x,[f"{r['height']}x{r['width']}\n{r['history_chunks']} history chunks" for r in rows]);ax.set_ylabel('Time / Dense FlashAttention (%)');ax.set_ylim(0,110);ax.set_title('Tile20: execution is ~20%, attention time remains ~28-30%')
 ax.legend(fontsize=9,loc='upper right',ncol=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 names={'_shared':'Sparse attention','_coarse':'Coarse score','_fine_age':'Fine score','_tiles':'Tile aggregation + select','_top':'Coarse top-k','_merge_restore':'Merge + query restore'};prof={z['name']:z['ms'] for z in largest['kernel_profile']};vals=[prof[k] for k in names]
 bars=bx.barh(list(names.values())[::-1],vals[::-1],color=['#aab5bf','#e59836','#e59836','#e59836','#e59836','#2468a2']);bx.bar_label(bars,fmt='%.2f ms',padding=4);bx.set_xlim(0,max(vals)*1.22);bx.set_xlabel('CUDA kernel time (ms; profiler sample)');bx.set_title('960x1664 / 6 history chunks / layer24: where time goes')
 fig.text(.5,.015,'Actual Dense-generated QKV; A100 40GB; layers 1/14/24/29. Operator test, not whole-video speed or quality.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(ROOT/'scaling.png',dpi=180);fig.savefig(ROOT/'scaling.pdf');plt.close(fig)

if __name__=='__main__':main()
