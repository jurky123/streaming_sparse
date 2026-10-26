import hashlib,json
from pathlib import Path
from .setup import ROOT,PARENT,CAPTURE

def main():
    cases={}
    for case in sorted(ROOT.glob('layer*_chunk*')):
        if not case.is_dir():continue
        record=json.loads((case/'results.json').read_text())
        for extra in ['low_candidates.json']:
            if (case/extra).exists():record['rows']+=json.loads((case/extra).read_text())['rows']
        assert json.loads((case/'status.json').read_text())['status']=='completed'
        cases[case.name]=record
    assert len(cases)==6
    validation=json.loads((ROOT/'validation.json').read_text())
    locked=json.loads((PARENT/'code_manifest.json').read_text())
    prev=ROOT.parent/'opt15_reorder_20261025'/'code_manifest.json'
    locked.update(json.loads(prev.read_text()))
    for f,digest in locked.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
    limitations=[
        'Six captured blocks from one Val trajectory at step3; no end-to-end video or independent-prompt quality claim.',
        'Layer24 trees calibrated on chunk1 and reused at chunk2/3; other layers calibrated on even token positions of the same captured event. These are diagnostic calibrations, not Train-only learned models.',
        'New algorithm: KV tokens are repaged, old prototypes are rebuilt, per-query top-k replaces threshold+81-action AV correction. It is not mask-preserving or equivalent to Opt14.',
        'Target budget is approximately 15%, not identical weighted token budgets: tail pages and ceil produce 15.29%-15.69% here; Opt14 budgets vary by layer.',
        'Opt14 timing covers routing plus attention with prepared prototypes; new warm timing also includes current KV regrouping and prototype reconstruction. Cold timing includes all new KV preparation. Neither is a full model/layer/video measurement.',
        'Frozen-all-KV time is an idealized diagnostic, not steady-state generation latency. Current KV must be rebuilt every step.',
        'Dense timings were captured separately; one run shows clock/timing variability. No claim of beating Dense from these measurements.',
        'Global rescue, actual-error joint fine-tuning, all-layer adapters, and automatic layer/budget selection are not implemented.',
        'Fine candidate scoring still pads P2/P4/P8 to16; K/V concatenation and repeated quantization remain; both are included in measured pipeline cost.',
    ]
    result=dict(status='implemented_validated_experimental',default_changed=False,cases=cases,validation=validation,limitations=limitations,parent_code_files_verified=len(locked))
    (ROOT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['Opt16 双侧分组路由：设计与实现结果','状态：实现完成；60 项验证通过；6 个真实 block / 33 项配置测试完成；未替换 Opt14 默认推理。','',
        '核心实现：响应距离校准二叉分桶树 → Q/KV 稳定排列 → KV token 重新组成页 → 保留 P16/P8/P4/P2 摘要 → 组级候选 → 逐 query 约15%精选 → INT8 QK/BF16 PV attention → Q 逆映射。',
        '组级候选测试25%/35%/50%/75%/100%；分组树默认64桶、深度6，运行时无聚类迭代。','',
        '35%候选结果（NMSE 相对 Dense；执行比例含 union 与 tile padding）：',
        'case                 Opt14 NMSE  新NMSE   Opt14 route+attn ms  新历史缓存/当前重建 ms  新全冷ms  执行比例']
    for name,c in cases.items():
        r=next(r for r in c['rows'] if r['variant']=='Both35')
        lines.append(f'{name:20s} {c["baseline"]["opt14_attention_nmse"]:10.5f} {r["attention_nmse_dense"]:8.5f} {r["opt14_route_attention_ms"]:18.3f} {r["history_cached_current_rebuilt_ms"]:23.3f} {r["cold_all_kv_ms"]:9.3f} {r["executed_fraction"]:8.2%}')
    c=cases['layer24_chunk3'];rows={r['variant']:r for r in c['rows']}
    lines+=['','第24层长历史预算扫描：','方案         NMSE     token保留  执行比例  route ms  attention ms  历史缓存/当前重建ms  全冷ms']
    for name in ['Identity1','Both1','Both75','Both50','Both35','Both25','Q50','K50']:
        r=rows[name];z=r['components_ms']
        lines.append(f'{name:10s} {r["attention_nmse_dense"]:.5f}  {r["keep_tokens"]:7.2%}  {r["executed_fraction"]:7.2%}  {z["route_only"]:8.3f} {z["attention_only"]:12.3f} {r["history_cached_current_rebuilt_ms"]:20.3f} {r["cold_all_kv_ms"]:8.3f}')
    r=rows['Both35'];oldbytes=r['memory']['old_weight_elements']*2;newbytes=r['memory']['new_fine_score_elements']*4
    lines+=['',f'该块 Q 分桶+稳定排序 {r["components_ms"]["q_assignment_and_order"]:.3f} ms。旧全量prototype权重约 {oldbytes/1e6:.1f} MB，新候选分数约 {newbytes/1e6:.1f} MB；这是对应中间量比较，不是峰值显存。',
        '观察：双侧分组+35%候选在部分层同时降低误差和时间；第29层误差退化，不能直接统一全层部署。',
        '仅做双侧分组、保留全候选会增加部分层 union；粗筛强度是实际计算预算的一部分。',
        '25%候选更快，但第24层短历史和第29层误差明显变大；没有将它默认启用。','',
        '计时边界：Opt14 不含摘要重建，新 warm 包含当前段重建；cold 含全部 KV 分组/摘要。所有数字均为单块 CUDA graph 回放，不是视频生成时延。',
        'Dense 计时存在单次运行间波动，未据此宣称超过 Dense。','',
        '限制与后续训练边界：']+[f'- {s}' for s in limitations]+['',
        '详细设计：DESIGN.txt；全部原始数据：report.json 与 layer*_chunk*/results.json / low_candidates.json。',
        '查看状态：python -m migration.unified_router.opt16.status',
        '验证：CUDA_VISIBLE_DEVICES=0 streaming_sparse/.venv/bin/python -m migration.runtime.bootstrap migration.unified_router.opt16.validate',
        '参考：SVG2 官方项目 https://svg-project.github.io/v2/ ；本实现不是其完整复现。']
    (ROOT/'report.txt').write_text('\n'.join(lines)+'\n')
    (ROOT/'parent_lock.json').write_text(json.dumps(locked,indent=2)+'\n')
    code={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    (ROOT/'code_manifest.json').write_text(json.dumps(code,indent=2)+'\n')
    (ROOT/'experiment.json').write_text(json.dumps(dict(status='completed_experimental_not_adopted',defaults_changed=False,captures=str(CAPTURE),parent=str(PARENT),cases=6,configurations=sum(len(c['rows'])for c in cases.values()),validation_cases=validation['total']),indent=2)+'\n')
    print('\n'.join(lines[:16]))

if __name__=='__main__':main()
