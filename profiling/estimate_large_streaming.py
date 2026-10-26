"""Extrapolate the measured 1.3B operation times to the Krea-Realtime-14B backbone.

Keeps the previous matched protocol (720p,3 history chunks,4 denoise+1 cache
write), rather than claiming to reproduce Krea's default server with KV recache,
fused projections and a different attention backend. No large-model timing is
measured here. Per-operation scaling assumes unchanged effective throughput.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    root = parser.parse_args().directory
    reference = json.loads((root/'router_cost_model.json').read_text())
    original = dict(layers=30, heads=12, dim=1536, ffn_dim=8960, head_dim=128)
    target = dict(layers=40, heads=40, dim=5120, ffn_dim=13824, head_dim=128)
    layer_ratio = target['layers']/original['layers']
    width_ratio = target['dim']/original['dim']
    ffn_ratio = target['ffn_dim']/original['ffn_dim']
    pair_ratio = target['layers']*target['heads']/(original['layers']*original['heads'])
    kernels = list(csv.DictReader((root/'exclusive_kernels.csv').open()))
    reference_parts = defaultdict(float)
    for row in kernels:
        if int(row['chunk']) >= 5:
            reference_parts[row['part']] += float(row['gpu_kernel_ms'])/23
    projections = {'self.q_projection','self.k_projection','self.v_projection',
                   'self.output_projection','cross.q_projection','cross.k_projection',
                   'cross.v_projection','cross.output_projection'}
    metadata = {'self.cache_metadata','self.total','cross.total','ffn.total','block.total'}
    # Small mixed embedding operations use their combined arithmetic ratio.
    embedding_scales = {
        'model.time_embedding': (256*5120+5120**2)/(256*1536+1536**2),
        'model.text_embedding': (4096*5120+5120**2)/(4096*1536+1536**2),
        'model.time_projection': width_ratio**2,
    }
    parts = []
    for part, ms in reference_parts.items():
        if part in projections:
            factor, law = layer_ratio*width_ratio**2, 'L*d^2'
        elif part in {'ffn.linear_in','ffn.linear_out'}:
            factor, law = layer_ratio*width_ratio*ffn_ratio, 'L*d*f'
        elif part == 'ffn.gelu':
            factor, law = layer_ratio*ffn_ratio, 'L*f'
        elif part in metadata:
            factor, law = layer_ratio, 'L (metadata/launch residual)'
        elif part.startswith(('self.', 'cross.', 'block.')):
            factor, law = layer_ratio*width_ratio, 'L*d (fixed Q/K/head_dim)'
        elif part in embedding_scales:
            factor, law = embedding_scales[part], 'mixed linear arithmetic, no L factor'
        elif part in {'model.head','model.patch_embedding'}:
            factor, law = width_ratio, 'd'
        else:
            assert part in {'chunk.total','model.total'}, part
            factor, law = 1., 'fixed small scheduler/model residual'
        parts.append(dict(part=part, reference_ms=ms, scale=factor, scaling_law=law,
                          projected_ms=ms*factor))
    wall_residual = reference['baseline']['chunk_generation_ms']-sum(reference_parts.values())
    baseline = sum(row['projected_ms'] for row in parts)+wall_residual
    module = sum(row['projected_ms'] for row in parts if row['part'].startswith('self.'))
    vae = reference['baseline']['vae_ms']
    rows, scopes = [], {}
    for calls in (4, 5):
        key = 'all30_four_denoise_context_dense' if calls == 4 else 'all30_all5_including_context'
        source = reference['scopes'][key]
        core = source['eligible_self_core_ms']*pair_ratio
        scopes[str(calls)] = dict(sparse_calls=calls, projected_eligible_core_ms=core,
                                 router_matrix_TFLOP=source['router_matrix_flops']*pair_ratio/1e12,
                                 projected_dense_QK_PV_TFLOP=source['dense_QK_PV_flops']*pair_ratio/1e12)
        for keep in (.1, .2):
            fraction = .25+.75*keep
            saved = core*(1-fraction)
            for model, source_ms in source['router_matrix_times_ms'].items():
                router_ms = source_ms*pair_ratio
                sparse_ms = baseline-saved+router_ms
                rows.append(dict(model='Krea-Realtime-14B backbone, matched protocol',
                                 sparse_calls=calls, history_keep=keep, router_model=model,
                                 projected_dense_chunk_ms=baseline,
                                 projected_dense_all5_self_module_ms=module,
                                 projected_eligible_core_ms=core,
                                 router_matrix_ms=router_ms,
                                 projected_core_plus_router_speedup=core/(fraction*core+router_ms),
                                 projected_all5_self_module_speedup=module/(module-saved+router_ms),
                                 projected_chunk_speedup=baseline/sparse_ms,
                                 projected_chunk_ms=sparse_ms,
                                 projected_chunk_plus_vae_speedup=(baseline+vae)/(sparse_ms+vae),
                                 additional_total_overhead_break_even_ms=saved,
                                 unmodeled_extra_overhead_ms=0.))
    cache_bytes = 2*target['layers']*43200*target['heads']*target['head_dim']*2
    notes = [
        'Analytical extrapolation, not measured Krea-Realtime-14B latency or quality.',
        'Matched protocol:720p1280x720,27chunks,3 historical chunks+full current,3latent frames/chunk,4denoise+1current clean-cache write;all layers/heads sparse hypothetically.',
        'Krea official server recomputes historical KV per block, defaults to5denoise calls, fuses QKV, and selects another attention backend. Native-server timings cannot be obtained from this simple scaling.',
        'Router unchanged means same per-query algorithm,page64,8prototypes,MLPs and FP32 with TF32disabled; all1600layer/head pairs still require routing. It does not mean fixed total router cost or that trained36pair weights can directly transfer.',
        'Current FP32 graph scenario is a compute-only upper bound on speedup within this model; any nonlinear,selection,mask,gather,launch and memory cost makes it lower. Algebraic graph changes invalidate its FLOP count.',
        'BF16 scenarios are counterfactual precision changes, retained only to compare with previous theoretical estimates; not the requested strictly unchanged router.',
        'Operation time scales by arithmetic/tensor size with same per-operation effective throughput; larger GEMMs, kernel fusion, memory bandwidth and parallelism can change that throughput.',
        'Clean wall minus exclusive kernel residual is held constant; no negative residual removed.',
        'Dense and sparse base use same rescaled primitive costs; sparse core linear in visible KV; no trained large-model sparse quality result exists here.',
        'A10040GB cannot hold dense BF16 14B weights plus this full KV cache simultaneously. No offload or communication modeled. A larger-memory GPU or distributed implementation changes latency.',
    ]
    provenance_file = root/'krea_source/provenance.json'
    report = dict(reference_model=original, target_model=target,
                  target_name='Krea Realtime14B',
                  sources=['https://github.com/krea-ai/realtime-video',
                           'https://github.com/krea-ai/realtime-video/blob/main/wan/configs/wan_t2v_14B.py'],
                  source_snapshot=json.loads(provenance_file.read_text()) if provenance_file.exists() else None,
                  pair_count_scale=pair_ratio,
                  projected_baseline=dict(chunk_ms=baseline, self_module_ms=module,
                                          unscaled_wall_residual_ms=wall_residual,
                                          same_vae_ms=vae),
                  memory=dict(dense_kv_bytes=cache_bytes, dense_kv_GiB=cache_bytes/2**30,
                              nominal_14B_BF16_weight_GiB=14e9*2/2**30),
                  formulas=dict(chunk='T/(T-C*(1-r)+R)',
                                router_scale='(40*40)/(30*12)=4.4444',
                                work_fraction='r=0.25+0.75*history_keep'),
                  assumptions=notes, scopes=scopes, rows=rows)
    (root/'large_model_estimates.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(root/'large_model_estimates.csv',rows)
    write_csv(root/'large_model_operation_scaling.csv',parts)
    print(json.dumps(dict(baseline=report['projected_baseline'],scopes=scopes,
                          all5=[row for row in rows if row['sparse_calls']==5]),indent=2))


if __name__ == '__main__':
    main()
