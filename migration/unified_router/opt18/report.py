"""Aggregate continuation and actual-error/latency scans; no deployment changes."""
import csv,hashlib,json
from pathlib import Path
from .setup import ROOT

def pareto(rows,cost='warm_ms'):
    return sorted([r for r in rows if not any(o[cost]<=r[cost] and o['nmse']<=r['nmse'] and (o[cost]<r[cost] or o['nmse']<r['nmse'])for o in rows if o is not r)],key=lambda r:r[cost])

def main():
    m=json.loads((ROOT/'experiment.json').read_text());training=[];results=[];baselines={}
    for layer in m['layers']:
        for variant in m['variants']:
            folder=ROOT/f'training/layer{layer:02d}/{variant}'
            d=json.loads((folder/'completed.json').read_text());best=json.loads((folder/'best_validation.json').read_text());curve=json.loads((folder/'curve.json').read_text())
            training.append(dict(layer=layer,variant=variant,updates=d['updates'],best_step=best['updates'],before=d['initial'],after=d['best'],relative_improvement=1-d['best']/d['initial'],tail=curve[-6:]))
            frontier=ROOT/f'frontier/layer{layer:02d}/{variant}'
            if (frontier/'opt14_validation.json').exists():baselines[str(layer)]=json.loads((frontier/'opt14_validation.json').read_text())['nmse']
            for f in sorted((frontier/'configs').glob('*.json')):results.append(json.loads(f.read_text()))
    complete=len(list((ROOT/'frontier').glob('layer*/*/completed.json')))==8
    fronts={str(layer):{cost:pareto([r for r in results if r['layer']==layer],cost)for cost in ['warm_ms','cold_ms']}for layer in m['layers']}
    limitations=[
        'Accuracy is over the existing96 Val events;128 fixed original queries/event supervise the error, full4680 queries determine hard grouping/quantization. Test remains unused.',
        'Timing uses six complete blocks per configuration: two Val prompts, three history lengths, step3. These are CUDA graph device timings, not model/video latency.',
        'Warm includes current KV rebuild; cold includes all KV rebuild and raw concatenation. Dense includes KV concatenation. Opt14 timings exclude prototype preparation and are a favorable reference.',
        'Pareto points use measured means, not uncertainty bounds. Small latency differences should not be treated as stable wins.',
        'Reported Pareto lists compare sparse configurations only. Dense is the zero-error reference and must also be considered; a sparse-only frontier is not evidence of beating Dense.',
        'Fresh64/128/256 trees are calibrated on Train using current projections; they are not gradient-fine-tuned, and differ from the trained best64 trees.',
        'All new configurations retain approximately15% KV. Candidate fraction and executed tile fraction are distinct metrics.',
        'No all29-layer deployment change or end-to-end video evaluation has occurred in this experiment.',
    ]
    dense_comparison={str(layer):dict(configs=sum(r['layer']==layer for r in results),warm_mean_faster_than_paired_dense=sum(r['layer']==layer and r['warm_ms']<r['dense_concat_ms']for r in results))for layer in m['layers']}
    summary=dict(status='completed'if complete else'partial',training=training,frontier_configuration_count=len(results),opt14_validation_nmse=baselines,pareto=fronts,dense_comparison=dense_comparison,limitations=limitations)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    columns=['layer','variant','tree','depth','fraction','reps','nmse','keep','executed','warm_ms','cold_ms','dense_concat_ms','opt14_prepared_route_attention_ms','q_grouping_ms','key']
    with (ROOT/'frontier.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    lines=['Opt18 续训与计算/精度实测','状态：'+summary['status']+f'；8组续训完成；已记录{len(results)}/140个配置。','',
        '固定best分组树继续训练，最终保留约15%；下面是相对于Opt17 best的额外收益。',
        'layer variant      Opt17 best  Opt18 best   额外改善  新best步数']
    for r in training:lines.append(f'{r["layer"]:5d} {r["variant"]:10s} {r["before"]:.6f}   {r["after"]:.6f}    {r["relative_improvement"]:6.2%}   {r["best_step"]}')
    lines+=['','原训练分组(best64)、16代表的候选预算扫描：','layer variant    候选   NMSE     执行比例  warm ms  cold ms  Dense ms  Opt14准备后ms']
    for r in sorted((r for r in results if r['tree']=='best64'and r['reps']==16),key=lambda r:(r['layer'],r['variant'],r['fraction'])):
        lines.append(f'{r["layer"]:5d} {r["variant"]:10s} {r["fraction"]:4.0%} {r["nmse"]:.6f}  {r["executed"]:7.2%} {r["warm_ms"]:8.3f} {r["cold_ms"]:8.3f} {r["dense_concat_ms"]:8.3f} {r["opt14_prepared_route_attention_ms"]:12.3f}')
    lines+=['','Quality50分组桶对比（16代表，候选50%；fresh代表新校准、未微调）：','layer tree           NMSE     warm ms  cold ms  Q分组ms  执行比例']
    for r in sorted((r for r in results if r['variant']=='Quality50'and r['reps']==16 and r['fraction']==.5),key=lambda r:(r['layer'],r['tree'])):
        lines.append(f'{r["layer"]:5d} {r["tree"]:14s} {r["nmse"]:.6f} {r["warm_ms"]:8.3f} {r["cold_ms"]:8.3f} {r["q_grouping_ms"]:8.3f} {r["executed"]:8.2%}')
    lines+=['','各层warm时延/实际误差的观测Pareto点（仅在稀疏配置内部比较；Dense仍须纳入决策；非部署选择）：']
    for layer in m['layers']:
        lines.append(f'Layer{layer}，Opt14相同Val NMSE={baselines.get(str(layer))}')
        for r in fronts[str(layer)]['warm_ms']:
            lines.append(f'  {r["variant"]} {r["key"]}: NMSE={r["nmse"]:.6f}, warm={r["warm_ms"]:.3f}ms, cold={r["cold_ms"]:.3f}ms')
    lines+=['','与配对Dense时延比较（6个完整块平均）：']
    for layer,d in dense_comparison.items():lines.append(f'Layer{layer}: {d["warm_mean_faster_than_paired_dense"]}/{d["configs"]} 个稀疏配置的warm均值快于Dense。')
    lines+=['','边界：']+['- '+s for s in limitations]
    (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    if results:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,layer in zip(axes.flat,m['layers']):
            rr=[r for r in results if r['layer']==layer]
            for name,color,marker in [('Fast35','tab:blue','o'),('Quality50','tab:orange','s')]:
                ss=[r for r in rr if r['variant']==name and r['tree']=='best64']
                ax.scatter([r['warm_ms']for r in ss],[r['nmse']for r in ss],label=name+' trained64',color=color,marker=marker,s=28)
            ss=[r for r in rr if r['tree']!='best64']
            if ss:ax.scatter([r['warm_ms']for r in ss],[r['nmse']for r in ss],label='fresh64/128/256',color='tab:green',marker='x')
            pp=fronts[str(layer)]['warm_ms'];ax.plot([r['warm_ms']for r in pp],[r['nmse']for r in pp],color='gray',alpha=.5)
            if rr:ax.scatter([sum(r['dense_concat_ms']for r in rr)/len(rr)],[0.],marker='*',s=100,color='black',label='Dense reference (NMSE=0)')
            if str(layer)in baselines:ax.axhline(baselines[str(layer)],color='tab:red',ls='--',label='Opt14 error')
            ax.set_title(f'Layer {layer}');ax.set_xlabel('History cached + current rebuild (ms)');ax.set_ylabel('Actual Val attention NMSE');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Opt18: measured block latency vs actual attention error');fig.tight_layout();fig.savefig(ROOT/'frontier.png',dpi=160);fig.savefig(ROOT/'frontier.pdf');plt.close(fig)
    print('\n'.join(lines[:14]))

if __name__=='__main__':main()
