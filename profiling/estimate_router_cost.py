"""Conditional router FLOP models, anchored to the measured dense steady chunks.

This is an analytical model, not a benchmark of an optimized sparse kernel.
Run: streaming_sparse/.venv/bin/python profiling/estimate_router_cost.py RESULTS_DIRECTORY
"""
import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from pathlib import Path


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(root, report, literature):
    """Keep authors' measurements separate from the conditional local cost model."""
    methods = literature['methods']
    flattened = []
    for method in methods:
        for measurement in method['measurements']:
            row = dict(method=method['method'], version=method['version'],
                       category=method['category'], backbone=method['backbone'],
                       resolution=method['resolution'], hardware=method['hardware'],
                       performance_scope=method['performance_scope'], **measurement,
                       evidence=method['evidence'], source_url=method['source_url'],
                       caveat=method['caveat'])
            for unit in ('ms', 's'):
                if f'baseline_{unit}' in measurement:
                    row['computed_speedup'] = measurement[f'baseline_{unit}']/measurement[f'method_{unit}']
            if 'baseline_fps' in measurement:
                row['computed_speedup'] = measurement['method_fps']/measurement['baseline_fps']
            flattened.append(row)
    columns = list(dict.fromkeys(key for row in flattened for key in row))
    write_csv(root/'streaming_sparse_literature.csv', [{key: row.get(key, '') for key in columns} for row in flattened])
    (root/'streaming_sparse_literature.json').write_text(json.dumps(literature, ensure_ascii=False, indent=2)+'\n')
    esc = lambda value: html.escape(str(value))
    pieces = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>720p Router 成本模型与流式稀疏注意力文献</title>
<style>body{max-width:1180px;margin:38px auto;padding:0 24px;font:16px/1.65 system-ui;color:#18202b;background:#fafbfc}h1,h2,h3{line-height:1.3}section{background:white;padding:22px;margin:20px 0;border:1px solid #d8dee4;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}td,th{padding:9px;border:1px solid #d8dee4;text-align:left;vertical-align:top}th{background:#eef3f8}a{color:#0969da}code{font-size:14px;overflow-wrap:anywhere}img{width:100%}.note{border-left:4px solid #bc4c00;padding-left:16px}.scroll{overflow-x:auto}</style>
<h1>720p Router 成本模型与流式稀疏注意力文献</h1>
<p>查询截至 2026-10-22。文献为作者报告值，本地 router 为条件性理论估计，尚未测量高效实现的运行时间。</p>
<p><a href="router_cost_model.json">成本模型 JSON</a> · <a href="router_scenarios.csv">加速比 CSV</a> · <a href="router_flops.csv">FLOP 拆分 CSV</a> · <a href="streaming_sparse_literature.csv">文献数据 CSV</a> · <a href="streaming_sparse_literature.json">文献与来源 JSON</a> · <a href="report.html">原始 profile 报告</a></p>
<section><h2>本地条件与计算范围</h2>
<p>Self-Forcing1.3B；1280×720；A100-SXM4-40GB；27 chunks；稳态取 chunk5–27。每 chunk 为3个 latent frames，10800 query tokens，最多32400历史 tokens，当前 chunk 完整保留。</p>
<p>“全 step”分别列四次 denoise（clean-cache write 仍 dense）与全部五次 forward（clean-cache write 也 sparse）。后者偏离现有训练/评估协议，需要另行验证视频质量。</p>''']
    b = report['baseline']
    pieces.append(f'<p>实测稳态生成 {b["chunk_generation_ms"]:.3f} ms/chunk；self-attention 整模块 {b["all5_self_module_ms"]:.3f} ms；VAE另计 {b["vae_ms"]:.3f} ms。投影、QK norm、RoPE、KV更新与O投影包含在 self-attention 整模块。</p>')
    pieces.append('''<p><code>r=(10800+k×32400)/43200=0.25+0.75k</code>；<code>T_sparse=T−C_attn×(1−r)+R_total</code>。R_total包括所有新增开销，表中仅估算矩阵乘法部分。</p></section>
<section><h2>现有 router 的矩阵运算量</h2>
<p>Page64，共507历史页，每页8原型，head_dim128。每个 query 独立路由。当前 chunk 的完整 QK 和加权内容聚合均重新计算，不能仅按少量原型点积估计。</p>
<p><code>每 head/forward: 8Q B P D + 4Q C D + 2Q B(112+576)</code>，另加每 chunk 构建一次 sketch。FP64元素运算、非线性、归一化、Top-K、mask、gather、访存和launch成本尚未计入。</p><table><tr><th>范围</th><th>Router矩阵TFLOP/chunk</th><th>占dense QK+PV FLOP比例</th><th>BF16峰值时间</th><th>同dense有效吞吐假设</th><th>当前FP32计算图峰值下界</th></tr>''')
    for name, scope in report['scopes'].items():
        times = scope['router_matrix_times_ms']
        pieces.append(f'<tr><td>{scope["calls"]}次 sparse forward</td><td>{scope["router_matrix_flops"]/1e12:.3f}</td><td>{scope["router_to_dense_matrix_flops"]:.2%}</td><td>{times["BF16_peak_312TF_optimistic_ceiling"]:.1f} ms</td><td>{times["BF16_same_effective_throughput_assumption"]:.1f} ms</td><td>≥{times["FP32_current_graph_19p5TF_peak_bound"]:.1f} ms</td></tr>')
    pieces.append('''</table><p class="note">FP32下界只适用于现有矩阵计算图及TF32关闭条件。改变算法、复用计算或改精度后须重算。BF16假设需要重写算子并验证数值/质量；312TF/s为硬件峰值，约206.5TF/s为dense QK/PV算术有效吞吐，都不是实测router速度。</p>
<p><a href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf">NVIDIA A100官方规格</a>：FP32 19.5TFLOPS，BF16 dense Tensor Core 312TFLOPS；未使用结构化2:4稀疏的624TFLOPS。</p></section>
<section><h2>全部五次 forward 稀疏：条件性加速比</h2>
<p>Keep10/20%均指历史保留率，当前 chunk完整保留；对应总attention密度32.5%/40%。所有值假设sparse core随保留KV数线性缩放，未计其他新增开销。</p>
<table><tr><th>Router假设</th><th>历史保留</th><th>Core+router加速</th><th>完整self-attn+router加速</th><th>生成chunk加速</th><th>生成+VAE加速</th><th>Chunk时间</th></tr>''')
    labels = {'zero_router_reference': '零开销参考', 'BF16_peak_312TF_optimistic_ceiling': 'BF16硬件峰值，乐观上限', 'BF16_same_effective_throughput_assumption': 'BF16与dense相同有效吞吐假设', 'FP32_current_graph_19p5TF_peak_bound': '现有FP32图硬件峰值，速度上界'}
    for row in report['rows']:
        if row['scope'] != 'all30_all5_including_context':
            continue
        pieces.append(f'<tr><td>{labels[row["router_model"]]}</td><td>{row["history_keep"]:.0%}</td><td>{row["eligible_core_plus_router_speedup"]:.3f}×</td><td>{row["all5_self_module_plus_router_speedup"]:.3f}×</td><td>{row["chunk_generation_speedup"]:.3f}×</td><td>{row["chunk_plus_vae_speedup"]:.3f}×</td><td>{row["chunk_generation_ms"]:.1f} ms</td></tr>')
    pieces.append('''</table><p>全部五次调用的新增总开销必须低于1405.3ms（Keep10%）/1249.1ms（Keep20%）才有正收益。若只做四次denoise稀疏，在同有效吞吐假设下生成chunk约1.112×/1.068×。</p>
<img src="router_sensitivity.png" alt="Router overhead sensitivity"><p><a href="router_sensitivity.pdf">下载PDF图</a></p></section>
<section><h2>文献数据：按范围核对</h2><p>同一行的kernel与E2E结果可能对应不同形状/密度；请结合每个设置说明。未找到完全匹配本地720p/A100/3历史/27chunk/全层全step的公开测试。</p>''')
    for method in methods:
        pieces.append(f'<h3><a href="{esc(method["source_url"])}">{esc(method["method"])}</a></h3>')
        for key in ('version', 'category', 'backbone', 'resolution', 'hardware', 'selection', 'performance_scope'):
            pieces.append(f'<p><strong>{esc(key)}</strong>：{esc(method[key])}</p>')
        pieces.append('<table><tr><th>设置</th><th>原始数据</th><th>加速比</th></tr>')
        for row in (r for r in flattened if r['method'] == method['method']):
            data = '; '.join(f'{key}={row[key]}' for key in ('baseline_ms', 'method_ms', 'baseline_s', 'method_s', 'baseline_fps', 'method_fps', 'baseline_vbench', 'method_vbench') if key in row)
            speed = row.get('computed_speedup', row.get('reported_speedup'))
            pieces.append(f'<tr><td>{esc(row["setting"])}</td><td>{esc(data)}</td><td>{speed:.4f}×</td></tr>')
        pieces.append(f'</table><p class="note">{esc(method["caveat"])}</p><p>证据位置：{esc(method["evidence"])}</p>')
        for key in ('quality', 'router_breakdown_setting', 'router_breakdown_percent', 'router_overhead'):
            if key in method:
                pieces.append(f'<p>{esc(key)}：{esc(method[key])}</p>')
    pieces.append('</section><section><h2>比较约束</h2><ul>')
    for rule in literature['comparison_rules']:
        pieces.append(f'<li>{esc(rule)}</li>')
    pieces.append('</ul></section></html>')
    (root/'router_and_literature.html').write_text('\n'.join(pieces))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    metadata = json.loads((root / 'metadata.json').read_text())
    chunks = list(csv.DictReader((root / 'chunk_summary.csv').open()))
    first = metadata['max_history_tokens'] // metadata['query_tokens'] + 2
    steady = [row for row in chunks if int(row['chunk']) >= first]
    kernels = list(csv.DictReader((root / 'exclusive_kernels.csv').open()))
    baseline = statistics.mean(float(row['baseline_wall_ms_median']) for row in steady)
    vae = statistics.mean(float(row['vae_stream_decode_ms']) for row in steady)

    def measured(core=False, denoise_only=False):
        return sum(float(row['gpu_kernel_ms']) for row in kernels
                   if int(row['chunk']) >= first
                   and (not denoise_only or row['stage'] == 'denoise')
                   and (row['part'] == 'self.sdpa_fused' if core
                        else row['part'].startswith('self.'))) / len(steady)

    q = current = metadata['query_tokens']
    history = metadata['max_history_tokens']
    dim, prototypes, page_size, layers, heads = 128, 8, 64, 30, 12
    pages = math.ceil(history / page_size)
    padded_history = pages * page_size
    # FMA counts as 2 FLOPs. Bias, nonlinear, reductions and data movement omitted.
    terms = [
        ('prototype_logits', '2*Q*B*P*D', 2*q*pages*prototypes*dim),
        ('directional_variance', '2*Q*B*P*D', 2*q*pages*prototypes*dim),
        ('prototype_content_mu', '2*Q*B*P*D', 2*q*pages*prototypes*dim),
        ('current_exact_QK', '2*Q*C*D', 2*q*current*dim),
        ('current_weighted_content', '2*Q*C*D', 2*q*current*dim),
        ('residual_content_projection', '2*Q*B*D*8', 2*q*pages*dim*8),
        ('calibrator_6_16_1', '2*Q*B*(6*16+16)', 2*q*pages*(6*16+16)),
        ('residual_head_17_32_1', '2*Q*B*(17*32+32)', 2*q*pages*(17*32+32)),
    ]
    score_per_head = sum(term[2] for term in terms)
    # Assignment and weighted K first/second moments, once per chunk/layer/head.
    build_per_head = 6 * padded_history * dim * prototypes
    extra_v_build_per_head = 2 * padded_history * dim * prototypes
    dense_per_head = 4 * q * (history + current) * dim
    module_all = measured()
    rows, scopes, flop_rows = [], {}, []
    for calls in (4, 5):
        scope = 'all30_four_denoise_context_dense' if calls == 4 else 'all30_all5_including_context'
        core = measured(core=True, denoise_only=calls == 4)
        eligible_module = measured(denoise_only=calls == 4)
        head_calls = layers * heads * calls
        router_flops = score_per_head * head_calls + build_per_head * layers * heads
        dense_flops = dense_per_head * head_calls
        effective_tflops = dense_flops / (core / 1000) / 1e12
        latencies = {
            'zero_router_reference': 0.,
            'BF16_peak_312TF_optimistic_ceiling': router_flops / 312e12 * 1000,
            'BF16_same_effective_throughput_assumption': router_flops / (effective_tflops * 1e12) * 1000,
            'FP32_current_graph_19p5TF_peak_bound': router_flops / 19.5e12 * 1000,
        }
        scopes[scope] = dict(calls=calls, eligible_self_core_ms=core,
                             eligible_self_module_ms=eligible_module,
                             dense_QK_PV_flops=dense_flops,
                             dense_effective_tflops=effective_tflops,
                             router_matrix_flops=router_flops,
                             router_to_dense_matrix_flops=router_flops/dense_flops,
                             router_matrix_times_ms=latencies,
                             V_branch_extra_build_flops=extra_v_build_per_head*layers*heads)
        for name, formula, flops in terms:
            flop_rows.append(dict(scope=scope, operation=name, formula=formula,
                                  frequency='per_layer_head_forward',
                                  per_instance_flops=flops, instances=head_calls,
                                  total_TFLOP=flops*head_calls/1e12))
        flop_rows.append(dict(scope=scope, operation='K_sketch_build', formula='6*H_padded*D*P',
                              frequency='once_per_chunk_layer_head',
                              per_instance_flops=build_per_head, instances=layers*heads,
                              total_TFLOP=build_per_head*layers*heads/1e12))
        for keep in (.1, .2):
            fraction = (current + keep*history)/(current+history)
            saved = core*(1-fraction)
            for model, router_ms in latencies.items():
                target = baseline-saved+router_ms
                rows.append(dict(scope=scope, history_keep=keep,
                                 total_attention_work_fraction=fraction, router_model=model,
                                 router_matrix_ms=router_ms, unmodeled_overhead_ms=0.,
                                 eligible_core_plus_router_speedup=core/(core*fraction+router_ms),
                                 eligible_self_module_plus_router_speedup=eligible_module/(eligible_module-saved+router_ms),
                                 all5_self_module_plus_router_speedup=module_all/(module_all-saved+router_ms),
                                 chunk_generation_ms=target,
                                 chunk_generation_speedup=baseline/target,
                                 chunk_plus_vae_speedup=(baseline+vae)/(target+vae),
                                 router_and_other_overhead_break_even_ms=saved,
                                 additional_overhead_budget_ms=saved-router_ms))
    source = Path(__file__).resolve().parents[1] / 'streaming_sparse/v08/router.py'
    report = dict(
        type='Analytical scenarios; no sparse runtime measured',
        baseline=dict(chunk_generation_ms=baseline, vae_ms=vae, all5_self_module_ms=module_all,
                      steady_chunks=[first, len(chunks)], hardware=metadata['gpu']),
        dimensions=dict(Q=q, C=current, H=history, H_padded=padded_history, B=pages,
                        D=dim, P=prototypes, page_size=page_size, layers=layers, heads=heads),
        router_source=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        formulas=dict(work_fraction='r=(C+k*H)/(C+H)',
                      router_matrix_per_head_forward='8*Q*B*P*D+4*Q*C*D+2*Q*B*(112+576)',
                      chunk='S=T/(T-C_attn*(1-r)+R_total)',
                      self_module='S=M/(M-C_attn*(1-r)+R_total)',
                      core_plus_router='S=C_attn/(r*C_attn+R_total)'),
        assumptions=[
            'K branch model; V branch has the same score matrix FLOPs and one extra small V sketch build.',
            'All 30 layers and 12 heads are hypothetical; current trained router has 36 pairs for three layers.',
            'Historical sketches built once per chunk and reused across calls; no rebuild per denoise step.',
            'Original FP32 GEMMs with TF32 disabled use the conditional 19.5 TFLOPS bound. This is not an algorithmic lower bound after algebraic rewrites or precision changes.',
            'BF16 scenarios assume redesigned Tensor Core GEMMs; no numerical or video-quality equivalence has been established.',
            'Same-effective-throughput uses measured dense BF16 QK/PV arithmetic throughput, not a measured router throughput. Narrow router GEMMs can be much less efficient.',
            'Sparse core time assumed linear in retained KV; unchanged full current chunk; history keep10/20 percent means delete90/80 percent.',
            'Top-K, mask generation, gather, exp/log, normalization, FP64 elementwise work, HBM traffic, launches and sparse-kernel inefficiency are unmodeled, nonnegative overheads.',
            'Inference-only soft BudgetGate/auxiliary training objectives omitted; exact hard selection is still needed.',
            'No reuse between router current-QK/content and final attention is assumed; a redesigned fused algebra can change the FLOP model.',
            'All5 sparsifies the clean-cache forward as well; this departs from the current dense cache-write training/evaluation protocol and requires quality validation.',
            'T is clean wall time; C/M are exclusive CUDA times from a separate profile with less than1 percent overhead.',
        ],
        hardware_source='https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf',
        scopes=scopes, rows=rows)
    (root/'router_cost_model.json').write_text(json.dumps(report, indent=2)+'\n')
    write_csv(root/'router_scenarios.csv', rows)
    write_csv(root/'router_flops.csv', flop_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, calls in zip(axes, (4, 5)):
        scope = scopes['all30_four_denoise_context_dense' if calls == 4 else 'all30_all5_including_context']
        for keep, color in ((.1, '#0969da'), (.2, '#bc4c00')):
            fraction = (current+keep*history)/(current+history)
            overheads = [i*10 for i in range(181)]
            values = [baseline/(baseline-scope['eligible_self_core_ms']*(1-fraction)+r) for r in overheads]
            ax.plot(overheads, values, label=f'History keep {keep:.0%}', color=color)
        for label, key, color in [('BF16 peak', 'BF16_peak_312TF_optimistic_ceiling', '#777'),
                                   ('Same throughput', 'BF16_same_effective_throughput_assumption', '#986ee2')]:
            ax.axvline(scope['router_matrix_times_ms'][key], linestyle=':', color=color, label=label)
        ax.axhline(1, color='black', linewidth=.8)
        ax.set(title=f'{calls} sparse forwards / chunk', xlabel='Total extra overhead R (ms/chunk)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Steady generation chunk speedup (VAE excluded)')
    axes[1].legend(fontsize=8)
    fig.suptitle('720p A100 / 3 historical chunks / 30 layers; analytical sensitivity')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'router_sensitivity.{ext}', dpi=160)
    literature = json.loads(Path(__file__).with_name('streaming_sparse_literature.json').read_text())
    render_report(root, report, literature)
    print(json.dumps(dict(baseline=report['baseline'], scopes=scopes, rows=[r for r in rows if r['scope'].startswith('all30_all5')]), indent=2))


if __name__ == '__main__':
    main()
