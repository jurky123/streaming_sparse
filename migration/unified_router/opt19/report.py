import csv,hashlib,json,statistics
from pathlib import Path
from .probe import ROOT
from ..opt18.setup import ROOT as PARENT
from ..common import atomic_json,sha

def main():
    summaries=[];details=[]
    for layer in [1,14,24,29]:
        for variant in ['Fast35','Quality50']:
            path=ROOT/f'final/layer{layer:02d}_{variant}/result.json'
            doc=json.loads(path.read_text());assert len(doc['configs'])==7
            assert sha(doc['checkpoint'])==doc['checkpoint_sha256']
            c,r=(35,8) if variant=='Fast35' else(50,16)
            parent=json.loads((PARENT/f'frontier/layer{layer:02d}/{variant}/configs/best64_c{c:03d}_r{r:02d}.json').read_text())
            baseline=statistics.mean(x['nmse'] for x in doc['parent_rows'])
            assert abs(baseline-parent['nmse'])<1e-7
            for record in doc['configs']:
                name=record['name'];events=doc['rows'][name]
                for x in events:details.append(dict(layer=layer,variant=variant,config=name,**x))
                row=dict(layer=layer,variant=variant,**record,parent_nmse=baseline,parent_warm_ms=parent['warm_ms'],
                    parent_executed=parent['executed'],nmse_change_percent=(record['nmse']/baseline-1)*100,
                    dense_speedup=record['dense_ms']/record['warm_ms'],cold_dense_speedup=record['dense_ms']/record['cold_ms'],
                    parent_speedup=parent['warm_ms']/record['warm_ms'],
                    max_abs_nmse_delta_vs_preliminary=max(abs(x['nmse_delta_vs_preliminary']) for x in events))
                summaries.append(row)
    for filename,rows in [('summary.csv',summaries),('event_results.csv',details)]:
        with (ROOT/filename).open('w') as f:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    profiles={name:json.loads((ROOT/f'profile/{name}/result.json').read_text()) for name in ['native','shared']}
    manifest=json.loads((ROOT/'validation_source_manifest.json').read_text())
    for f,digest in manifest.items():assert sha(f)==digest,('Source changed',f)
    summary=dict(configurations=summaries,profiles=profiles,validation_events_per_configuration=96,
        validation_prompts=4,validation_seeds=[0,1],timed_events_per_configuration=6,
        numerical_source_and_parent_checkpoints_unchanged=True,video_e2e=False,training_performed=False,
        default_deployment_changed=False,limitations=['Only4pilotlayers','Val used for screening; not held-out Test','NMSE on128 positions with fullQ execution','warm assumes owned unchanged history; cold rebuilds it'])
    atomic_json(ROOT/'summary.json',summary)
    lines=['Opt19：tile预算与多模块执行优化','',
      '完成：8组权重×10个初筛配置；8组权重×7个最终组合；每个配置96个Val事件、full4680-query执行。',
      'Val为4个prompt×2个seed×3个历史长度×4个去噪step。误差沿用128采样query真实Dense输出NMSE。',
      '测速为2个prompt、seed0、step3、3种历史长度的6个block；每项80次预热、4轮交替30次重放。',
      'Warm包括当前KV分组/摘要/量化、Q重排、路由、attention及输出复原；Cold还重建全部历史分组、摘要、量化。',
      '下列执行率为QK/PV等效tile工作量，包含尾部padding；不代表全模型FLOPs。保留率按真实token加权。','',
      'layer variant    config          keep% exec% NMSE      delta%  warm_ms cold_ms Dense_ms speedup']
    for r in summaries:
        lines.append(f"{r['layer']:2d} {r['variant']:9} {r['name']:15} {100*r['keep']:5.2f} {100*r['executed']:5.2f} {r['nmse']:.6f} {r['nmse_change_percent']:+7.2f} {r['warm_ms']:7.3f} {r['cold_ms']:7.3f} {r['dense_ms']:7.3f} {r['dense_speedup']:6.2f}x")
    lines+=['','实施的优化：',
      '- 每个query仍独立计算候选概率，tile内求概率和选固定页预算；共享16/32个query的最终页列表。',
      '- 保留native32路径，单独衡量保持原逐query决策的执行优化。',
      '- 自有历史缓存：当前KV直接写入预分配区域，避免每步复制历史；历史量化K也缓存，只更新当前块。',
      '- 融合FP16 page prototype构建，FP32 softmax/count；投影与加权均值交换，省掉128D原型中间量。',
      '- 粗筛处理宽度64→128；精筛根据历史实际P2/P4/P8/P16分支，减少padding计算。',
      '- 共享tile直接使用页列表，省掉full query×page mask及union；复用原INT8 QK/BF16 PV与split merge。',
      '所有误差均以实际最终kernel输出计算；没有重新训练。','',
      '公共kernel复用结果：',
      '已接通并验证PyTorch2.6 FlexAttention完整块路径、FlashInfer0.2.8 FA2 paged-prefill。',
      '相同mask的输出NMSE很小，但当前A100/16或32-query shape上速度未超过所选INT8/BF16实现。',
      '这些公共kernel基准另排除了每chunk最后8-token尾页；最终选用路径支持尾页，不需要此近似。',
      'FlashInfer固定长度plan一次，在GPU更新索引并捕获run；不对每次变化的索引重新CPU plan。',
      'FlashInfer逻辑32-query tile内部会使用64-query CTA，因此不能用逻辑保留率冒充它的硬件执行率。',
      '完整记录：library_*.json，attention_probe.json，route_probe.json，probe_Fast35.json。','',
      '新的热点（层24、Fast35、同一长历史sample；kernel时间不含host gaps）：']
    for name,d in profiles.items():
        lines.append(name+' paired block times: '+json.dumps(d['timings']))
        lines.append(name+' modules: '+json.dumps(d['profile_module_kernel_ms']))
        for x in d['kernels'][:6]:lines.append('  '+json.dumps(x))
    lines+=['','未采用：tf32x3融合摘要虽误差极小但更慢；coarse_width256会改变部分候选；native16仅缩小并集，实测更慢。',
      '完整视频尚未评估；当前仅4个试验层，不宣称全模型加速或视频质量改善；默认Opt14未替换。',
      '复现：bash opt19_status.sh；python -m migration.unified_router.opt19.final_validate --layer 24 --variant Fast35',
      '使用仓库streaming_sparse/.venv/bin/python；公共FlashInfer基准需TORCH_CUDA_ARCH_LIST=8.0。',
      '来源与数据边界见PLAN.txt；代码/父checkpoint校验见validation_source_manifest.json。']
    (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    for row,variant in enumerate(['Fast35','Quality50']):
        for col,layer in enumerate([1,14,24,29]):
            ax=axes[row,col];rr=[r for r in summaries if r['variant']==variant and r['layer']==layer]
            native=next(r for r in rr if r['name']=='native32')
            ax.scatter([native['dense_ms']],[0],marker='*',s=90,c='black',label='Dense')
            ax.scatter([native['parent_warm_ms']],[native['parent_nmse']],marker='x',s=70,c='gray',label='Opt18')
            ax.scatter([native['warm_ms']],[native['nmse']],marker='s',s=45,c='green',label='Opt19 per-query')
            for g,color in [(16,'tab:blue'),(32,'tab:orange')]:
                points=sorted([r for r in rr if r['mode']=='shared' and r['group']==g],key=lambda r:r['budget'])
                ax.plot([r['warm_ms'] for r in points],[r['nmse'] for r in points],'-o',c=color,label=f'Shared Q{g}')
                for r in points:ax.annotate(f"{r['budget']:.0%}",(r['warm_ms'],r['nmse']),xytext=(3,5),textcoords='offset points',fontsize=8)
            ax.set_title(f'Layer {layer} / {variant}');ax.set_xlabel('Complete warm block (ms)');ax.set_ylabel('Attention output NMSE');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Opt19: measured compute / accuracy trade-offs (Val, not video E2E)')
    fig.savefig(ROOT/'tradeoff.png',dpi=160);fig.savefig(ROOT/'tradeoff.pdf');plt.close(fig)
    atomic_json(ROOT/'completion.json',dict(status='completed',preliminary_configurations=80,final_configurations=56,events_per_config=96,profiles=2,video_e2e=False,default_changed=False))
    print(ROOT/'REPORT.txt')

if __name__=='__main__':main()
