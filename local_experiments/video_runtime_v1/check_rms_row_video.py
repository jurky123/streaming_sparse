"""Same-process paired video timing with alternating order after warmup."""
import argparse
import json
import time
import torch
from .accepted_tiers import config
from .deployment import configure, load_weights, make_runtime
from .rms_mean import fused, row_fused
from .weights import ROOT
from ..broad_common import Engine
from ..rollout_suite import generate_chunk

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', choices=['base', 'intersection', 'step', 'dense'], required=True)
    parser.add_argument('--pairs', type=int, default=10, help='Number of repetitions for a single video.')
    parser.add_argument('--native-route-qk', action='store_true',
                        help='Compare native-input QK graphs while keeping the accepted RMS implementation.')
    parser.add_argument('--scale-in-qk', action='store_true', help='Also move logit scaling into the QK producer.')
    parser.add_argument('--single-av', action='store_true', help='Compare one versus two AV products in exact attention.')
    parser.add_argument('--group-direction', action='store_true')
    parser.add_argument('--paired-gather', action='store_true')
    parser.add_argument('--inline-u', action='store_true')
    parser.add_argument('--bundle',action='store_true',help='Combine inline Risk, paired KV gather and one-hot projection.')
    parser.add_argument('--batch-clean',type=int,default=0)
    parser.add_argument('--group-streams',action='store_true')
    parser.add_argument('--overlap-clean',action='store_true')
    parser.add_argument('--tail-merge',action='store_true')
    parser.add_argument('--multi-video', action='store_true', help='Test five prompts at seeds 703 and 704, one pair per combination.')
    parser.add_argument('--direct-output',action='store_true')
    parser.add_argument('--prompt', default='city_walk')
    parser.add_argument('--seed', type=int, default=701)
    parser.add_argument('--reuse-selected-ids',action='store_true')
    parser.add_argument('--tie-direct',action='store_true')
    parser.add_argument('--resident-major',action='store_true')
    parser.add_argument('--contiguous-commit',action='store_true')
    parser.add_argument('--rounded-ffn',action='store_true')
    parser.add_argument('--ffn-group-commit',action='store_true')
    parser.add_argument('--raw-group',action='store_true')
    parser.add_argument('--norm-raw',action='store_true')
    parser.add_argument('--partial-page',action='store_true')
    parser.add_argument('--large-group',action='store_true')
    parser.add_argument('--root-group',action='store_true')
    parser.add_argument('--group-features',action='store_true')
    parser.add_argument('--group-feature-bundle',action='store_true')
    parser.add_argument('--projected-inputs',action='store_true')
    parser.add_argument('--rms-rope',action='store_true')
    parser.add_argument('--matched-risk',action='store_true')
    parser.add_argument('--fixed-pages',action='store_true')
    parser.add_argument('--page-route',action='store_true')
    parser.add_argument('--cuda-pool',action='store_true')
    parser.add_argument('--rechecked-mass',action='store_true')
    parser.add_argument('--partial-mass',action='store_true')
    parser.add_argument('--risk-bound',action='store_true')
    parser.add_argument('--group-radix',action='store_true')
    parser.add_argument('--group-sort',action='store_true')
    parser.add_argument('--strided-tail',action='store_true')
    parser.add_argument('--residual-norm',action='store_true')
    parser.add_argument('--repeated-context',action='store_true')
    parser.add_argument('--radix-select',action='store_true')
    parser.add_argument('--channel-major-stats',action='store_true')
    parser.add_argument('--resident-route',action='store_true')
    parser.add_argument('--summary-window',action='store_true')
    parser.add_argument('--cached-projection',action='store_true')
    parser.add_argument('--fused-exact-tail',action='store_true')
    args = parser.parse_args()
    if args.pairs < 1 and not args.multi_video:
        parser.error("--pairs must be positive")
    if args.rechecked_mass:args.partial_mass=True
    if args.group_radix:args.group_sort=True
    if args.bundle:
        args.paired_gather=args.inline_u=True
    if args.scale_in_qk:
        args.native_route_qk = True
    if args.native_route_qk and args.tier == 'dense':
        parser.error('Native route QK requires a sparse tier')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dense = args.tier == 'dense'
    cfg = config('base' if dense else args.tier, local_assets='/tmp/streaming_sparse-0/runtime')
    if args.matched_risk or args.fixed_pages or args.page_route:cfg.page_route=False
    if args.contiguous_commit or args.rounded_ffn or args.ffn_group_commit:
        cfg.layernorm_mod=False
        cfg.exact_group_direction=False
        cfg.contiguous_commit=False
        cfg.rounded_ffn=False
    if args.raw_group:cfg.layernorm_mod=False
    if args.inline_u:cfg.inline_risk_projection=False
    if args.paired_gather or args.group_direction:cfg.group_gathers=False
    if args.tie_direct or args.direct_output or args.reuse_selected_ids:
        cfg.selected_tie_order=cfg.direct_output=False
    if args.batch_clean:cfg.clean_stream_width=0
    if args.tail_merge:cfg.tail_merge=False
    if args.direct_output:cfg.clean_stream_width,cfg.tail_merge=4,True
    if args.rounded_ffn or args.ffn_group_commit or args.raw_group:cfg.rounded_ffn=True
    if args.raw_group:cfg.contiguous_commit=True
    if args.residual_norm:cfg.resident_route=False
    if args.cached_projection:cfg.summary_window=False
    if args.summary_window:cfg.cached_projection=True
    if args.repeated_context:cfg.repeated_context=True
    configure(cfg)
    engine = Engine()
    engine.wrapper.requires_grad_(False)
    if dense:
        from .backbone_ops import install
        patches = install(engine, cfg)
        norm, reset = patches['rms_dataflow'], engine.reset
    else:
        model, reader, phi, basis, _ = load_weights(cfg)
        rt = make_runtime(engine, model, reader, phi, basis, cfg)
        norm, reset = rt.rms_dataflow, rt.reset
    cases=[(args.prompt,args.seed)] * args.pairs
    prompts=[args.prompt]
    if args.multi_video:
        from .validate_single_av import PROMPTS
        prompts=list(PROMPTS)
        cases=[(prompt,seed) for seed in (703,704) for prompt in prompts]
    embeddings={prompt:torch.load(ROOT / f'artifacts/e24_joint_representatives_v1/embeddings/{prompt}.pt',
                                  weights_only=True).cuda() for prompt in prompts}
    if args.group_direction or args.paired_gather:
        from . import grouping_dataflow as grouping
        implementations = [rt.group_graph, grouping.GraphGrouping(rt.config, spatial_softmax=True)]
    if args.inline_u:
        from .risk_projected import risk_inline
        risk_functions = (rt.batch_graph.risk_function, risk_inline)
        risk_entries = [{}, {}]
    if args.page_route or args.risk_bound or args.partial_mass:route_entries=[{},{}]
    if args.fixed_pages:
        from .risk_online import matched_risk
        rt.batch_graph.risk_function=matched_risk
        route_entries=[{},{}]
    if args.matched_risk:
        from .risk_online import matched_risk
        risk_functions=(rt.batch_graph.risk_function,matched_risk)
        risk_entries=[{},{}]
    if args.batch_clean:
        side=torch.cuda.Stream() if args.overlap_clean else None
        from .group_batch import BatchedGrouping
        if args.group_streams:
            from .group_batch import ParallelGrouping
            rt.batched_group=ParallelGrouping(rt.config,args.batch_clean)
        else:rt.batched_group=BatchedGrouping(rt.config)
    if args.resident_major:
        cache_layouts=[]
        for store in rt.stores.values():
            for name in ('k','v'):
                native=store.cache[name]
                b,n,h,d=native.shape
                major=native.view(b,h,n,d).transpose(1,2)
                cache_layouts.append((store.cache,name,(native,major)))
        rt.batch_graph.reuse_selected_ids='ties'
        rt.direct_output=True
    if args.reuse_selected_ids or args.tie_direct or args.projected_inputs:route_entries=[{},{}]
    if args.ffn_group_commit and not dense:
        from .group_batch import ParallelGrouping
        group_versions=[ParallelGrouping(rt.config,4,exact_small=v,fused_norm=v) for v in (False,True)]
    if (args.raw_group or args.norm_raw) and not dense:
        from .group_batch import ParallelGrouping
        group_versions=[ParallelGrouping(rt.config,4,exact_small=v,fused_norm=True) for v in (True,'raw')]
    if args.large_group or args.root_group or args.group_features or args.group_feature_bundle:
        from .group_batch import ParallelGrouping
        if args.group_features or args.group_feature_bundle:
            group_versions=[
                ParallelGrouping(rt.config,4,exact_small='root',fused_norm=True,
                                 fused_features=enabled,staged_softmax=enabled and args.group_feature_bundle)
                for enabled in (False,True)]
        else:
            modes=('raw','root' if args.root_group else 'large')
            group_versions=[ParallelGrouping(rt.config,4,exact_small=mode,fused_norm=True) for mode in modes]
    if args.partial_page or args.cuda_pool:
        from functools import partial
        from .summary_moments import build
        summary_versions=(build,partial(build,partial_page=args.partial_page,cuda_pool=args.cuda_pool))
    if args.radix_select:route_entries=[{},{}]
    if args.channel_major_stats:
        from . import risk_partial_mass
        route_entries=[{},{}]
    rows = []
    for pair in range(-1, len(cases)):
        prompt,seed=cases[max(0,pair)]
        emb=embeddings[prompt]
        sample = dict(prompt=prompt,seed=seed)
        outputs = {}
        counts = {}
        for which in ((0, 1) if pair % 2 == 0 else (1, 0)):
            if args.group_direction or args.paired_gather:
                grouping.FUSED_SMALL_DIRECTION = bool(which) and args.group_direction
                grouping.PAIRED_GATHER = bool(which) and args.paired_gather
                grouping.ONE_HOT_GATHER = bool(which) and args.bundle
                rt.group_graph = implementations[which]
            if args.tail_merge:
                rt.workspace.tail_merge=bool(which)
            if args.residual_norm:
                (patches['pointwise_blocks'] if dense else rt.pointwise_blocks).residual_norm=bool(which)
            elif args.repeated_context:
                (patches['repeated_context'] if dense else rt.repeated_context).enabled=bool(which)
            elif args.radix_select:
                rt.batch_graph.radix_select=bool(which)
                rt.batch_graph.entries=route_entries[which]
            elif args.channel_major_stats:
                risk_partial_mass.CHANNEL_MAJOR_STATS=bool(which)
                rt.batch_graph.entries=route_entries[which]
            elif args.resident_route:
                rt.resident_route=bool(which)
            elif args.summary_window:
                rt.summary_window=bool(which)
            elif args.cached_projection:
                from .rms_mean import set_cached_projection
                set_cached_projection(engine,bool(which))
                if not dense:rt.cached_projection=bool(which)
            elif args.fused_exact_tail:
                rt.workspace.fused_exact_tail=bool(which)
            elif args.partial_mass:
                rt.batch_graph.partial_mass=('recheck' if args.rechecked_mass else True) if which else False
                rt.batch_graph.entries=route_entries[which]
            elif args.group_sort or args.strided_tail or args.risk_bound:
                if args.group_sort:
                    for group in rt.batched_group.groupers:
                        group.fused_sort=('radix' if args.group_radix else True) if which else False
                if args.strided_tail:
                    rt.workspace.strided_tail=bool(which)
                if args.risk_bound:
                    rt.batch_graph.direct_risk_bound=bool(which)
                    rt.batch_graph.entries=route_entries[which]
            elif args.page_route:
                rt.batch_graph.fixed_page_sets=rt.batch_graph.matched_probabilities=bool(which)
                rt.batch_graph.entries=route_entries[which]
            elif args.fixed_pages:
                rt.batch_graph.fixed_page_sets=bool(which)
                rt.batch_graph.entries=route_entries[which]
            elif args.rms_rope:
                from .rms_mean import set_rotary
                set_rotary(engine,bool(which))
            elif args.projected_inputs:
                rt.batch_graph.projected_route_inputs=bool(which)
                rt.batch_graph.entries=route_entries[which]
            elif args.batch_clean:
                rt.clean_group_batch=args.batch_clean if which else 0
                rt.commit_stream=side if which else None
            elif args.large_group or args.root_group or args.group_features or args.group_feature_bundle:rt.batched_group=group_versions[which]
            elif args.partial_page or args.cuda_pool:rt.summary_graph=summary_versions[which]
            elif args.norm_raw:
                (patches['pointwise_blocks'] if dense else rt.pointwise_blocks).fuse_norm=bool(which)
                if not dense:rt.batched_group=group_versions[which]
            elif args.raw_group:rt.batched_group=group_versions[which]
            elif args.ffn_group_commit:
                (patches['rounded_ffn'] if dense else rt.rounded_ffn).enabled=bool(which)
                if not dense:
                    rt.batched_group=group_versions[which]
                    for store in rt.stores.values():store.contiguous_commit=bool(which)
            elif args.rounded_ffn:
                patch=patches['rounded_ffn'] if dense else rt.rounded_ffn
                patch.enabled=bool(which)
            elif args.contiguous_commit:
                for store in rt.stores.values():store.contiguous_commit=bool(which)
            elif args.resident_major:
                for cache,name,layouts in cache_layouts:cache[name]=layouts[which]
            elif args.reuse_selected_ids or args.tie_direct:
                if args.tie_direct:rt.direct_output=bool(which)
                rt.batch_graph.reuse_selected_ids='ties' if args.tie_direct and which else bool(which)
                rt.batch_graph.entries=route_entries[which]
            elif args.direct_output:
                rt.direct_output=bool(which)
            elif args.inline_u or args.matched_risk:
                rt.batch_graph.risk_function = risk_functions[which]
                rt.batch_graph.entries = risk_entries[which]
            elif args.single_av:
                from . import legacy_components
                legacy_components.SINGLE_AV = bool(which)
            elif args.native_route_qk:
                # The BF16 / FP32 entry dtype distinguishes the two CUDA graphs.
                rt.batch_graph.native_route_qk = bool(which)
                rt.batch_graph.scale_in_qk = bool(which) and args.scale_in_qk
            elif not (args.group_direction or args.paired_gather or args.tail_merge):
                norm.function = (fused, row_fused)[which]
            torch.cuda.synchronize()
            start = time.perf_counter()
            reset()
            z = torch.cat([generate_chunk(engine, emb, seed, c) for c in range(9)], dim=1)
            torch.cuda.synchronize()
            sample[('old_s','new_s')[which]] = time.perf_counter() - start
            outputs[which] = z.cpu()
            counts[which] = [] if dense else rt.counters.tolist()
        a, b = outputs[0], outputs[1]
        sample.update(bitwise=torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
                      max_abs=(a.float()-b.float()).abs().max().item(), counters=counts)
        if args.partial_mass or args.fused_exact_tail or args.cached_projection or args.summary_window:
            delta=a.float()-b.float()
            sample['nmse']=float(delta.square().sum()/a.float().square().sum())
            sample['mae']=float(delta.abs().mean())
            sample['chunk_nmse']=[float(x.square().sum()/y.float().square().sum()) for x,y in zip(delta.chunk(9,dim=1),a.chunk(9,dim=1))]
        print(pair, sample, flush=True)
        if pair >= 0:
            rows.append(sample)
            experiment = 'native_route_qk_v1' if args.native_route_qk else 'rms_row_v1'
            if args.scale_in_qk:
                experiment = 'scaled_route_qk_v1'
            if args.single_av:
                experiment = 'exact_head_major_v1'
            if args.group_direction:
                experiment = 'group_direction_v1'
            if args.paired_gather:
                experiment = 'group_gather_v1'
            if args.inline_u:
                experiment = 'risk_inline_u_v1'
            if args.bundle:
                experiment = 'risk_group_bundle_v1'
            if args.batch_clean:experiment=f'batch_clean_{args.batch_clean}_v1'
            if args.group_streams:experiment=f'parallel_clean_{args.batch_clean}_v1'
            if args.overlap_clean:experiment=f'overlap_clean_{args.batch_clean}_v1'
            if args.tail_merge:
                experiment='tail_merge_overlap_v1' if args.overlap_clean else 'tail_merge_v1'
            if args.direct_output:experiment='direct_output_v1'
            if args.reuse_selected_ids:experiment='reuse_selected_ids_v1'
            if args.tie_direct:experiment='tie_direct_v1'
            if args.resident_major:experiment='resident_head_major_v1'
            if args.contiguous_commit:experiment='contiguous_commit_v1'
            if args.rounded_ffn:experiment='ffn_rounded_gelu_v1'
            if args.ffn_group_commit:experiment='ffn_group_commit_v1'
            if args.raw_group:experiment='raw_group_direction_v1'
            if args.norm_raw:experiment='norm_raw_v1'
            if args.partial_mass:experiment='rechecked_mass_v1' if args.rechecked_mass else 'partial_mass_video_v1'
            if args.cuda_pool:experiment='cuda_pool_v1'
            if args.partial_page:experiment='partial_page_v1'
            if args.large_group:experiment='large_group_direction_v1'
            if args.root_group:experiment='root_group_direction_v1'
            if args.group_features:experiment='group_features_v1'
            if args.group_feature_bundle:experiment='group_feature_bundle_v1'
            if args.projected_inputs:experiment='projected_route_inputs_v1'
            if args.rms_rope:experiment='rms_rope_v1'
            if args.matched_risk:experiment='risk_matched_norm_v1'
            if args.fixed_pages:experiment='fixed_page_select_v1'
            if args.group_sort:experiment='group_sort_v1'
            if args.strided_tail:experiment='strided_tail_v1'
            if args.risk_bound:experiment='group_sort_tail_bound_v1' if args.group_sort and args.strided_tail else 'risk_bound_v1'
            if args.page_route:experiment='page_route_v1'
            if args.group_radix:experiment='group_radix_tail_bound_v1' if args.risk_bound and args.strided_tail else 'group_radix_v1'
            if args.residual_norm:experiment='residual_norm_v1'
            if args.repeated_context:experiment='repeated_context_v1/packed'
            if args.radix_select:experiment='radix_select_v1'
            if args.channel_major_stats:experiment='risk_variance_dot_v1'
            if args.resident_route:experiment='resident_route_v1'
            if args.summary_window:experiment='summary_window_v1'
            if args.cached_projection:experiment='cached_projection_v1'
            if args.fused_exact_tail:experiment='fused_exact_tail_v1'
            if args.multi_video:experiment+='/multi_video'
            if not args.multi_video and (args.prompt,args.seed)!=('city_walk',701):
                experiment+=f'/repeat_{args.prompt}_{args.seed}'
            path = ROOT / f'artifacts/video_runtime_v1/{experiment}/{args.tier}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(rows, indent=2))
            if (args.partial_mass or args.single_av or args.group_direction or args.bundle) and pair == 0:
                torch.save(outputs, path.with_suffix('.pt'))
    assert args.partial_mass or args.group_direction or args.single_av or args.scale_in_qk or all(row['bitwise'] and row['max_abs'] == 0 and
               row['counters'][0] == row['counters'][1] for row in rows)
    if dense:
        for patch in reversed(list(patches.values())):
            patch.close()
    else:
        rt.close()

if __name__ == '__main__':
    main()
