"""Produce reproducible CSV/JSON tables, standalone plots and an HTML profile report."""
from pathlib import Path
import argparse,csv,html,json,statistics,math
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from profile_streaming import csv_write,dump


def readcsv(path):
    return list(csv.DictReader(Path(path).open())) if Path(path).exists() else []

def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory).resolve()
    metadata=json.loads((root/'metadata.json').read_text());status=json.loads((root/'status.json').read_text())
    asset=json.loads((Path(__file__).resolve().parents[1]/'migration/backbone_audit.json').read_text())
    metadata['teacher_checkpoint']='checkpoints/self_forcing_dmd.pt'
    metadata['teacher_sha256_verified_by_native_loader']=asset['sha256']
    metadata['postrun_git_source_unchanged']=metadata['source_sha256']==metadata['source_sha256_after']
    dump(root/'metadata.json',metadata)
    assert status['status']=='completed',status
    clean=[json.loads(x.read_text()) for x in sorted(root.glob('clean_[0-9]*.json'))]
    core=json.loads((root/'core_0.json').read_text());detail=json.loads((root/'detail_0.json').read_text())
    exclusive=readcsv(root/'exclusive_kernels.csv');vae=readcsv(root/'vae_decode_chunks.csv');encoding=readcsv(root/'text_encoding.csv')
    layers=[4,14,24];n=metadata['latent_frames']//3
    def median(values):return statistics.median(values)
    def stage_times(run,c,stage):return sum(x['cuda_span_ms'] for x in run['forwards'] if x['chunk']==c and x['stage']==stage)
    def core_times(c,stage='denoise',selected=None,kind='self'):
        return sum(x['cuda_span_ms'] for x in core['core'] if x['chunk']==c and x['stage']==stage and x['kind']==kind and (selected is None or x['layer'] in selected))
    chunk=[]
    for c in range(1,n+1):
        rows=[next(x for x in run['chunks'] if x['chunk']==c) for run in clean]
        wall=[x['wall_ms'] for x in rows];gpu=[x['cuda_span_ms'] for x in rows]
        den=median([stage_times(r,c,'denoise') for r in clean]);context=median([stage_times(r,c,'context') for r in clean]);dec=next(x for x in vae if int(x['chunk'])==c)
        cr=next(x for x in core['chunks'] if x['chunk']==c)
        chunk.append(dict(chunk=c,history_chunks=min(c-1,metadata['max_history_tokens']//10800),history_tokens=min(c-1,metadata['max_history_tokens']//10800)*10800,visible_kv_tokens=(min(c-1,metadata['max_history_tokens']//10800)+1)*10800,
            baseline_wall_ms_median=median(wall),baseline_wall_ms_min=min(wall),baseline_wall_ms_max=max(wall),baseline_cuda_span_ms_median=median(gpu),denoise_four_forwards_ms=den,dense_context_forward_ms=context,chunk_other_cuda_span_ms=median(gpu)-den-context,
            self_sdpa_denoise_ms_core_probe=core_times(c),self_sdpa_context_ms_core_probe=core_times(c,'context'),cross_sdpa_denoise_ms_core_probe=core_times(c,kind='cross'),selected3_self_sdpa_denoise_ms_core_probe=core_times(c,selected=layers),core_probe_chunk_wall_ms=cr['wall_ms'],vae_stream_decode_ms=float(dec['wall_ms']),rgb_frames=int(dec['rgb_frames']),peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in rows)))
    csv_write(root/'chunk_summary.csv',chunk)
    # Exclusive kernel assignment prevents QKV, attention and enclosing blocks being double counted.
    module=defaultdict(float);per_chunk=defaultdict(lambda:defaultdict(float));per_layer=defaultdict(lambda:defaultdict(float))
    def group(part):
        if part=='self.sdpa_fused':return 'Self SDPA core'
        if part.startswith('self.rope'):return 'RoPE Q/K'
        if part.startswith('self.kv_cache') or part=='self.cache_metadata':return 'KV update/indexing'
        if part.startswith('self.'):return 'Self projections/norm/layout'
        if part.startswith('cross.'):return 'Cross attention'
        if part.startswith('ffn.'):return 'FFN'
        if part.startswith('block.'):return 'Block norms/modulation/residual'
        return 'Model/scheduler/other'
    layer_rows=[]
    for x in exclusive:
        c=int(x['chunk']);layer=int(x['layer']);ms=float(x['gpu_kernel_ms']);part=x['part']
        if c<1:continue
        module[part]+=ms;per_chunk[c][group(part)]+=ms
        per_layer[(layer,c)][part]+=ms
    totals=[dict(part=k,gpu_kernel_ms=v,share_of_generation_gpu_kernels=v/sum(module.values())) for k,v in sorted(module.items(),key=lambda x:-x[1])]
    csv_write(root/'module_summary.csv',totals)
    stability=[]
    steady_start=metadata['max_history_tokens']//10800+2
    metrics={'baseline_chunk_wall_ms':[x['baseline_wall_ms_median'] for x in chunk]}
    for name in sorted({g for c in per_chunk.values() for g in c}):
        metrics[name]=[per_chunk[c][name] for c in range(1,n+1)]
    for name,values in metrics.items():
        vals=values[steady_start-1:]
        if not vals:continue
        mean=statistics.mean(vals);sd=statistics.pstdev(vals)
        stability.append(dict(metric=name,steady_chunk_start=steady_start,steady_chunk_end=n,mean_ms=mean,std_ms=sd,coefficient_of_variation=sd/mean if mean else 0.,min_ms=min(vals),max_ms=max(vals),slope_ms_per_chunk=float(np.polyfit(np.arange(steady_start,n+1),vals,1)[0]) if len(vals)>1 else 0.))
    csv_write(root/'steady_state_stability.csv',stability)
    for (layer,c),parts in sorted(per_layer.items()):
        if layer<0:continue
        layer_rows.append(dict(chunk=c,layer=layer,total_gpu_kernel_ms=sum(parts.values()),self_sdpa_ms=parts.get('self.sdpa_fused',0),cross_sdpa_ms=parts.get('cross.sdpa_fused',0),rope_ms=parts.get('self.rope_q',0)+parts.get('self.rope_k',0),ffn_ms=sum(v for k,v in parts.items() if k.startswith('ffn.'))))
    csv_write(root/'layer_chunk_summary.csv',layer_rows)
    csv_write(root/'chunk_module_kernels.csv',[dict(chunk=c,**dict(parts)) for c,parts in sorted(per_chunk.items())])
    csv_write(root/'baseline_forward_events.csv',[dict(repeat=i,**x) for i,run in enumerate(clean) for x in run['forwards']])
    csv_write(root/'attention_core_events.csv',core['core'])
    count_by=defaultdict(int)
    for row in core['core']:count_by[(row['stage'],row['kind'])]+=1
    assert count_by[('denoise','self')]==n*4*30,count_by
    assert count_by[('context','self')]==n*30,count_by
    for r in core['core']:
        if r['kind']=='self':assert r['kv_tokens']==min(r['chunk'],metadata['max_history_tokens']//10800+1)*10800
    # Rooflines are models, not measured sparse speedups. Projection/current/context stay dense.
    theory=[]
    for label,selected in [('current3layers',layers),('all30layers_hypothetical',None)]:
        for keep in (0.,.1,.2,.3):
            rows=[]
            for x in chunk:
                t=x['core_probe_chunk_wall_ms'];saved=0
                for ev in core['core']:
                    if ev['chunk']!=x['chunk'] or ev['stage']!='denoise' or ev['kind']!='self' or (selected is not None and ev['layer'] not in selected):continue
                    h=max(0,ev['kv_tokens']-ev['q_tokens'])
                    saved+=ev['cuda_span_ms']*(1-keep)*h/ev['kv_tokens']
                rows.append(dict(scenario=label,history_keep=keep,chunk=x['chunk'],eligible_attention_ms=core_times(x['chunk'],selected=selected),assumed_saved_ms=saved,probe_baseline_chunk_ms=t,estimated_generation_speedup=t/(t-saved),estimated_chunk_plus_vae_speedup=(t+x['vae_stream_decode_ms'])/(t-saved+x['vae_stream_decode_ms']),assumptions='zero router/sketch/gather overhead; attention core time linear in KV tokens; current KV, context write, projections and all other work unchanged'))
            total=sum(x['probe_baseline_chunk_ms'] for x in rows);saved=sum(x['assumed_saved_ms'] for x in rows);decode=sum(x['vae_stream_decode_ms'] for x in chunk)
            theory.append(dict(scenario=label,history_keep=keep,estimated_generation_speedup=total/(total-saved),estimated_generation_plus_vae_speedup=(total+decode)/(total-saved+decode),baseline_ms=total,assumed_saved_ms=saved))
            csv_write(root/f'theory_chunks_{label}_keep{keep:.1f}.csv',rows)
    csv_write(root/'theoretical_speedup.csv',theory)
    # Attention algorithm work independent of QKV projections.
    work=[]
    for x in chunk:
        q=10800;k=x['visible_kv_tokens'];work.append(dict(chunk=x['chunk'],q_tokens=q,kv_tokens=k,qk_flops_per_layer_forward=2*12*q*k*128,pv_flops_per_layer_forward=2*12*q*k*128,softmax_elements_per_layer_forward=12*q*k,current_kv_fraction=q/k,history_kv_fraction=(k-q)/k,keep20_attention_flop_speedup=k/(q+.2*(k-q))))
    csv_write(root/'attention_work.csv',work)
    baseline=[x['total_wall_ms'] for x in clean];total=median(baseline);vg=sum(x['vae_stream_decode_ms'] for x in chunk)
    warm_text=median([float(x['wall_ms']) for x in encoding if x['cold']=='False'])
    cache=median([x['initialization']['wall_ms'] for x in clean]);full_est=total+vg+warm_text+cache
    summary=dict(metadata=metadata,measurement_repeats=len(clean),baseline_generation_ms_median=total,baseline_generation_ms_repeats=baseline,baseline_generation_ms_min=min(baseline),baseline_generation_ms_max=max(baseline),warm_text_encode_ms=warm_text,cache_initialize_ms=cache,streaming_vae_sum_ms=vg,warm_text_to_rgb_serial_estimate_ms=full_est,serial_estimate_note='Sum of separately measured phases with no overlap; not a measured combined execution or an MP4 write latency.',generation_plus_vae_fps_estimate=metadata['decoded_frames']/((total+vg)/1000),core_probe_wall_ms=core['total_wall_ms'],core_probe_overhead_percent=(core['total_wall_ms']/total-1)*100,detailed_profiler_wall_ms=detail['total_wall_ms'],detailed_profiler_overhead_percent=(detail['total_wall_ms']/total-1)*100,total_exclusive_generation_gpu_kernel_ms=sum(module.values()),core_probe_counts={str(k):v for k,v in count_by.items()},rolling_equivalence_passed=json.loads((root/'rolling_equivalence.json').read_text())['passed'],instrumentation_output_bitwise_equal=True,steady_state_stability=stability,profile_scope='one training prompt and one seed, repeated latency measurements; BF16 SDPA, no compilation, 3 history chunks plus1 current, no LoRA/router/sparsity enabled',theoretical_speedup=theory,limitations=['GPU kernel sums, CUDA elapsed spans and host wall times are different metrics and are kept separate.','Inclusive ranges overlap; only exclusive kernel categories are stacked or summed.','Fused attention does not expose individual QK/softmax/PV kernel times. Explicit diagnostic results are a different execution algorithm.','Theoretical speedups assume zero routing and data movement overhead and linear attention time scaling; they are not measured sparse throughput.','Three selected layers means original zero-based layers4/14/24, only four denoising passes; the fifth clean-cache pass remains dense.','Runtime rolling cache and 720p indexing adaptation preserve the requested visible window; native 480p hard-coded token indexing was not used.'])
    kernel_totals={'self_attention_all_parts_ms':sum(v for k,v in module.items() if k.startswith('self.')),'self_attention_fused_core_ms':module.get('self.sdpa_fused',0.),'cross_attention_all_parts_ms':sum(v for k,v in module.items() if k.startswith('cross.')),'ffn_ms':sum(v for k,v in module.items() if k.startswith('ffn.'))}
    kernel_totals['other_generation_kernels_ms']=sum(module.values())-kernel_totals['self_attention_all_parts_ms']-kernel_totals['cross_attention_all_parts_ms']-kernel_totals['ffn_ms']
    summary['nonoverlapping_gpu_kernel_groups']=kernel_totals
    scales=[]
    for kv in sorted({x['kv_tokens'] for x in core['core'] if x['kind']=='self'}):
        vals=[x['cuda_span_ms'] for x in core['core'] if x['kind']=='self' and x['stage']=='denoise' and x['kv_tokens']==kv]
        scales.append(dict(visible_kv_tokens=kv,self_sdpa_single_layer_ms_median=median(vals),sample_count=len(vals)))
    xx=np.array([x['visible_kv_tokens'] for x in scales]);yy=np.array([x['self_sdpa_single_layer_ms_median'] for x in scales]);slope,intercept=np.polyfit(xx,yy,1);pred=slope*xx+intercept
    fit=dict(slope_ms_per_kv_token=float(slope),intercept_ms=float(intercept),r_squared=float(1-((yy-pred)**2).sum()/((yy-yy.mean())**2).sum()),q_tokens=10800,rows=scales,scope='Low-overhead event spans for actual denoising self-SDPA; different layers/steps pooled by key length; fit is diagnostic, not a sparse implementation.')
    dump(root/'attention_scaling_fit.json',fit);csv_write(root/'attention_scaling_by_kv.csv',scales);summary['attention_scaling_fit']=fit
    dump(root/'summary.json',summary)
    # Exportable figures with English labels for portable fonts.
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(name):
        plt.tight_layout();plt.savefig(root/(name+'.png'),bbox_inches='tight');plt.savefig(root/(name+'.pdf'),bbox_inches='tight');plt.close();figures.append(name)
    xs=np.arange(1,n+1)
    fig,ax=plt.subplots(figsize=(12,4));med=np.array([r['baseline_wall_ms_median']/1000 for r in chunk]);lo=np.array([r['baseline_wall_ms_min']/1000 for r in chunk]);hi=np.array([r['baseline_wall_ms_max']/1000 for r in chunk]);ax.plot(xs,med,'o-',label='Transformer streaming chunk wall');ax.fill_between(xs,lo,hi,alpha=.15,label='Min/max across clean repeats');ax.plot(xs,[x['denoise_four_forwards_ms']/1000 for x in chunk],label='Four denoising forwards (CUDA spans)');ax.plot(xs,[x['dense_context_forward_ms']/1000 for x in chunk],label='Dense clean-cache write (CUDA span)');ax.plot(xs,[x['vae_stream_decode_ms']/1000 for x in chunk],label='VAE stream decode (separate pass)');ax.axvline(4,color='gray',ls=':',label='3-history window full');ax.set(xlabel='Chunk (1-based)',ylabel='Seconds',title='True 1280x720 streaming: 27 chunks, 3 history chunks');ax.legend(fontsize=8,ncol=2);save('chunk_latency')
    groups=['Self SDPA core','Self projections/norm/layout','RoPE Q/K','KV update/indexing','Cross attention','FFN','Block norms/modulation/residual','Model/scheduler/other']
    fig,ax=plt.subplots(figsize=(12,5));ys=[np.array([per_chunk[c][g]/1000 for c in xs]) for g in groups];ax.stackplot(xs,*ys,labels=groups);ax.set(xlabel='Chunk',ylabel='Exclusive GPU kernel seconds',title='Non-overlapping module kernel breakdown (instrumented run)');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8);save('chunk_module_breakdown')
    selfgroups={'Q/K/V projection + norms':['self.q_projection','self.k_projection','self.v_projection','self.q_norm','self.k_norm'],'RoPE Q':['self.rope_q'],'RoPE K':['self.rope_k'],'KV update + metadata':['self.kv_cache_update','self.cache_metadata'],'Fused QK-softmax-PV':['self.sdpa_fused'],'Output projection':['self.output_projection'],'Layout / other':['self.total','self.attention_backend']}
    bypart=defaultdict(lambda:defaultdict(float))
    for x in exclusive:
        if int(x['chunk'])>0:bypart[int(x['chunk'])][x['part']]+=float(x['gpu_kernel_ms'])
    fig,ax=plt.subplots(figsize=(12,5));ax.stackplot(xs,*[[sum(bypart[c][p] for p in ps)/1000 for c in xs] for ps in selfgroups.values()],labels=list(selfgroups));ax.set(xlabel='Chunk',ylabel='Exclusive GPU kernel seconds',title='Self-attention internal operations (all30 layers; denoise + context)');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8);save('self_attention_parts')
    steps=np.zeros((5,n))
    for c in range(1,n+1):
        for step in range(4):steps[step,c-1]=median([sum(x['cuda_span_ms'] for x in r['forwards'] if x['chunk']==c and x['stage']=='denoise' and x['step']==step) for r in clean])/1000
        steps[4,c-1]=chunk[c-1]['dense_context_forward_ms']/1000
    fig,ax=plt.subplots(figsize=(12,3));im=ax.imshow(steps,aspect='auto',origin='upper',extent=(.5,n+.5,4.5,-.5));ax.set(xlabel='Chunk',yticks=range(5),yticklabels=['Denoise0','Denoise1','Denoise2','Denoise3','Dense context'],title='Native forward duration by step and chunk');fig.colorbar(im,ax=ax,label='CUDA span seconds');save('step_chunk_heatmap')
    heat=np.zeros((30,n))
    for ev in core['core']:
        if ev['stage']=='denoise' and ev['kind']=='self':heat[ev['layer'],ev['chunk']-1]+=ev['cuda_span_ms']
    fig,ax=plt.subplots(figsize=(12,5));im=ax.imshow(heat,aspect='auto',extent=(.5,n+.5,29.5,-.5));ax.set(xlabel='Chunk',ylabel='Original layer index',title='Denoising self-SDPA: layer x chunk (4 steps summed)');fig.colorbar(im,ax=ax,label='CUDA elapsed ms');save('layer_chunk_attention')
    fig,ax=plt.subplots(figsize=(12,4))
    for label,style in [('current3layers','o-'),('all30layers_hypothetical','s--')]:
        r=readcsv(root/f'theory_chunks_{label}_keep0.2.csv');ax.plot(xs,[float(x['estimated_generation_speedup']) for x in r],style,label=label+' (keep20% history)')
    ax.axhline(1,color='gray',lw=.8);ax.set(xlabel='Chunk',ylabel='Estimated transformer speedup',title='Amdahl model: zero routing overhead, attention time linear in KV length');ax.legend();save('theoretical_speedup')
    diagnostic=readcsv(root/'attention_parts_diagnostic.csv')
    if diagnostic:
        cs=sorted(set(int(x['chunk']) for x in diagnostic));keys=['qk_ms','scale_ms','softmax_fp32_ms','probability_cast_ms','pv_ms'];bottom=np.zeros(len(cs));fig,ax=plt.subplots(figsize=(9,4))
        for key in keys:
            vals=[median([float(x[key]) for x in diagnostic if int(x['chunk'])==c]) for c in cs];ax.bar(np.array(cs)-.16,vals,width=.32,bottom=bottom,label=key.removesuffix('_ms'));bottom+=vals
        vals=[median([float(x['fused_full_query_ms_median']) for x in diagnostic if int(x['chunk'])==c]) for c in cs];ax.bar(np.array(cs)+.16,vals,width=.32,label='Actual fused SDPA (separate microbench)');ax.set(xlabel='Chunk shape',ylabel='Milliseconds',title='DIAGNOSTIC: materialized tiled vs fused attention; different algorithms');ax.legend(fontsize=8);save('attention_parts_diagnostic')
    def table(rows,keys,labels=None,precision=3):
        def cell(v):return f'{v:.{precision}f}' if isinstance(v,float) else str(v)
        return '<table><thead><tr>'+''.join('<th>'+html.escape((labels or {}).get(k,k))+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(cell(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table>'
    selected_theory=[x for x in theory if x['history_keep']==.2]
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>720p 原始流式生成 Profile</title><style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1400px;margin:36px auto;padding:0 24px;color:#142234}}h1,h2{{line-height:1.3}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{padding:18px;background:#edf3fa;border-radius:10px;min-width:220px}}.value{{font-size:27px;font-weight:700}}img{{width:100%;max-width:1260px}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:16px 0}}td,th{{border:1px solid #d7dfe9;padding:6px;text-align:right}}th{{background:#edf3fa}}.scroll{{overflow:auto}}.note{{background:#fff4d9;padding:16px;border-left:4px solid #e5ad2a}}code{{background:#eef2f6;padding:2px 4px}}a{{color:#1763a4}}</style><h1>720p 原始流式生成：27 chunk 详尽 Profile</h1><p>Self-Forcing 1.3B · 单张 A100-SXM4-40GB · BF16 · batch1 · 真正 1280×720 · 27×3 latent 帧 / 321 RGB 帧 · 最多3个历史 chunk + 当前 chunk · 四步去噪 + 一次 Dense clean-cache 写入。</p><div class="cards"><div class="card">干净基准 Transformer 总时间<div class="value">{total/1000:.3f} s</div>{len(clean)} 次同 prompt / seed 重复的中位数</div><div class="card">逐 chunk VAE 解码时间之和<div class="value">{vg/1000:.3f} s</div>独立解码 pass；无视频编码/写盘</div><div class="card">细粒度 profiler 增加的耗时<div class="value">{summary['detailed_profiler_overhead_percent']:+.2f}%</div>核心探针：{summary['core_probe_overhead_percent']:+.2f}%</div></div><div class="note"><strong>度量边界：</strong>干净运行用于真实延迟；细粒度 profiler 用于独占 GPU kernel 归因；二者不混加。CUDA span 含流上的等待和调度间隙，kernel 时间之和只包含设备执行。QK/softmax/PV 已融合，显式分解只是诊断基准，不能当作融合内核内部的分项时间。理论稀疏加速比是零路由成本、core 时间随可见 KV 长度线性缩放的估算，不是已实现或实测加速。</div><h2>运行协议与正确性</h2><p>窗口总共43,200个可见 token，其中当前10,800个全保留、历史最多32,400个。chunk1/2/3/4 的可见 KV 分别为10,800/21,600/32,400/43,200，之后保持不变。滚动 cache 与追加存储的最近窗口计算，使用原始第14层、真实720p形状，在27个 chunk 上各进行两次写入并逐次比对，输出完全一致；干净运行、core probe 和细粒度 profiler 的生成 latent 也逐 bit 一致。原科学源码身份保持不变。</p><p>GPU kernel trace 后端、每个事件的算子名与 kernel 名可在 operators.csv / kernel_events.jsonl.gz / trace.json.gz 中核查。Prompt: <code>{html.escape(metadata['prompt_id'])}</code>；seed {metadata['seed']}。</p>'''
    for name in figures:body+=f'<h2>{html.escape(name.replace("_"," "))}</h2><a href="{name}.pdf">PDF</a><br><img src="{name}.png">'
    body+='<h2>全27个 chunk 的原始汇总</h2><p>所有时间单位为 ms。baseline 为干净重复中位数；core_probe 为独立低开销探针运行。</p><div class="scroll">'+table(chunk,['chunk','history_tokens','visible_kv_tokens','baseline_wall_ms_median','baseline_wall_ms_min','baseline_wall_ms_max','denoise_four_forwards_ms','dense_context_forward_ms','chunk_other_cuda_span_ms','self_sdpa_denoise_ms_core_probe','self_sdpa_context_ms_core_probe','selected3_self_sdpa_denoise_ms_core_probe','vae_stream_decode_ms','peak_allocated_GiB'])+'</div>'
    body+='<h2>固定窗口后的稳定性（chunk5 起）</h2><p>chunk4 首次填满3个历史 chunk；chunk5起执行滚动驱逐。此表区分填满窗口与实际滚动维护的影响。</p>'+table(stability,['metric','steady_chunk_start','steady_chunk_end','mean_ms','std_ms','coefficient_of_variation','slope_ms_per_chunk'],precision=5)
    body+='<h2>独占 GPU kernel 分项总计</h2>'+table(totals,['part','gpu_kernel_ms','share_of_generation_gpu_kernels'],precision=5)
    body+='<h2>历史 Keep20% 的理论估算</h2>'+table(selected_theory,['scenario','history_keep','estimated_generation_speedup','estimated_generation_plus_vae_speedup','assumed_saved_ms'],precision=5)+'<p>current3layers 仅指原编号4/14/24层的去噪 self-attention；all30layers 是假设对照。四步去噪全部执行，context写入、QKV/O投影、RoPE、FFN、cross-attention和VAE保持原样。路由、原型/Sketch、索引压缩、数据搬运及稀疏 kernel 效率损失均未计入，真实加速通常更低。</p>'
    body+='<h2>机器可读数据</h2><ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['summary.json','metadata.json','chunk_summary.csv','baseline_forward_events.csv','attention_core_events.csv','chunk_module_kernels.csv','module_summary.csv','layer_chunk_summary.csv','exclusive_kernels.csv','inclusive_ranges.csv','operators.csv','attention_work.csv','attention_scaling_fit.json','attention_scaling_by_kv.csv','attention_parts_diagnostic.csv','theoretical_speedup.csv','steady_state_stability.csv','vae_decode_chunks.csv','text_encoding.csv','rolling_equivalence.json','gpu_telemetry.csv','kernel_events.jsonl.gz','trace.json.gz'] if (root/name).exists())+'</ul><h2>解释边界</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['limitations'])+'</ul><h2>复现</h2><pre>CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 streaming_sparse/.venv/bin/python profiling/profile_streaming.py --output profiling/results/reproduce_720p --chunks 27 --history-chunks 3 --repeats 3\nstreaming_sparse/.venv/bin/python profiling/analyze_streaming.py profiling/results/reproduce_720p</pre></html>'
    (root/'report.html').write_text(body)
    print(json.dumps({k:v for k,v in summary.items() if k in ('baseline_generation_ms_median','streaming_vae_sum_ms','core_probe_overhead_percent','detailed_profiler_overhead_percent','theoretical_speedup')},indent=2))

if __name__=='__main__':main()
