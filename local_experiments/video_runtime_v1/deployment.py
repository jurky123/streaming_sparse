"""Shared accepted deployment configuration for benchmark and resident workers."""
from functools import partial
from types import SimpleNamespace
from pathlib import Path
import os
from .weights import ROOT,load

CURRENT=dict(aligned_coefficients=False,page_prefetch=False,column_mass=False,anchor_scale=False,repeated_context=True,radix_select=True,channel_major_stats=True,residual_norm=True,resident_route=True,summary_window=True,cached_projection=True,fused_exact_tail=True,rechecked_mass=True,group_radix=True,strided_tail=True,direct_risk_bound=True,cuda_pool=True,page_route=True,rms_rope=True,projected_route_inputs=True,group_feature_bundle=True,layernorm_mod=True,rounded_ffn=True,contiguous_commit=True,exact_group_direction='root',selected_tie_order=True,direct_output=True,clean_stream_width=4,tail_merge=True,inline_risk_projection=True,group_gathers=True,scale_in_qk=True,native_route_qk=True,rms_row=True,risk_online=False,summary_moments=True,native_rope=True,exact_fold=False,qk_tile64=True,rms_mean=True,rms_dataflow=True,block_pointwise=True,grouping_spatial=True,grouping_softmax=False,grouping_dataflow=True,mode='stream',full_pset=True,pset_steps='all',pset_policy=None,route_m16=True,reader_vector_g2=False,model_seed=17,checkpoint=str(ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt'),reader_checkpoint=str(ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt'),tile_batch=72,history_pages=360,stream_tail_mode=4,stream_warps=8,bf16_qk=True,split_summary=True,fused_risk=False,static_route=False,moment_scores=True,late_reader=True,blocked_scores=True,packed_scores=True,reader_page32=False,lazy_scores=False,reader_split=True,reader_layout=True,reader_layout_kind='a',scan_exclude=True,projected_risk=True,cached_logmass=True,legacy_components='rope,attention',profile=False,validate=False,local_assets=None,reader_output32=True,coeff_bf16=True,endpoint_den=True,cached_endpoints=True,scan_fp32=True)

def current(**overrides):return SimpleNamespace(**(CURRENT|overrides))
def identity(a):
    result={k:getattr(a,k,v) for k,v in CURRENT.items() if k!='local_assets'}
    if result['fused_exact_tail']:
        from ..learned_cache.io import digest
        result['exact_kernel_sha']=digest(ROOT/'local_experiments/gap_exact_round_kernel.py')
        result['tail_kernel_sha']=digest(ROOT/'local_experiments/review_v1/tuned_attention.py')
    if result['page_prefetch']:
        from ..learned_cache.io import digest
        result['page_prefetch_sha']=digest(Path(__file__).parent/'page_prefetch.py')
    if result['repeated_context']:
        from ..learned_cache.io import digest
        result['native_cross_attention_sha']=digest(ROOT/'wan/modules/model.py')
        result['repeated_context_kernel_sha']=digest(Path(__file__).parent/'csrc/repeated_context.cu')
    if result['rms_rope']:
        from ..learned_cache.io import digest
        result['native_self_attention_sha']=digest(ROOT/'wan/modules/causal_model.py')
    if result['pset_policy']:
        from ..learned_cache.io import digest
        result['pset_policy_sha']=digest(Path(result['pset_policy']))
        result['static_route']=True
        result['pset_steps']='all'
    if result['group_radix']:
        from ..learned_cache.io import digest
        result['group_radix_source_sha']=digest(Path(__file__).parent/'csrc/group_radix.cu')
    if result['summary_moments'] and result['cuda_pool']:
        from ..learned_cache.io import digest
        result['cuda_pool_source_sha']=digest(Path(__file__).parent/'csrc/pool_representative.cu')
    if result['grouping_spatial']:
        from ..learned_cache.io import digest
        result['grouping_spatial_source_sha']=digest(Path(__file__).parent/'csrc/group_softmax.cu')
    if result['rms_dataflow'] and result['rms_mean']:
        from ..learned_cache.io import digest
        result['rms_mean_source_sha']=digest(Path(__file__).parent/'csrc/rms_mean.cu')
    return result
def configure(a):
    from . import risk_partial_mass
    risk_partial_mass.CHANNEL_MAJOR_STATS=getattr(a,"channel_major_stats",False)
    from . import legacy_components
    legacy_components.EXACT_FOLD=getattr(a,'exact_fold',CURRENT['exact_fold'])
    from . import tensor_bmm
    tensor_bmm.QK_TILE64=getattr(a,'qk_tile64',CURRENT['qk_tile64'])
    if getattr(a,'grouping_spatial',False) and getattr(a,'grouping_softmax',False):raise ValueError('Choose stable spatial or experimental contiguous softmax, not both')
    from . import scores,reader_reduce,scores_moments,pipeline_graph,runtime
    scores.SCAN_EXCLUDE=a.scan_exclude
    reader_reduce.PAGE32=a.reader_page32;reader_reduce.SPLIT=a.reader_split;reader_reduce.LAYOUT=a.reader_layout or a.reader_split;reader_reduce.LAYOUT_KIND=a.reader_layout_kind
    reader_reduce.VECTOR_GROUP=2 if getattr(a,'reader_vector_g2',False) else 1
    reader_reduce.WARPS=4;reader_reduce.SHARD=128;reader_reduce.AGG32=True;reader_reduce.PACKED=a.packed_scores;reader_reduce.MOMENTS=a.moment_scores
    reader_reduce.SCAN_FP32=getattr(a,'scan_fp32',False)
    reader_reduce.OUTPUT32=getattr(a,'reader_output32',False);reader_reduce.ENDPOINT_DEN=getattr(a,'endpoint_den',False)
    from . import endpoint_den
    endpoint_den.CACHED_GROUPS=getattr(a,'cached_endpoints',False)
    scores_moments.BLOCKED=a.blocked_scores;scores_moments.LAZY=a.lazy_scores
    if a.late_reader:
        from .reader_late import reader_kernel
        reader_reduce.reader_kernel=reader_kernel
    scorer=scores.grouped_scores if a.mode=='baseline' else scores.fused_grouped_scores
    pipeline_graph.grouped_scores=scorer;runtime.grouped_scores=scorer
    if a.local_assets:
        from streaming_sparse.v08 import backbone
        original=backbone.protocol
        def local_protocol():
            p=dict(original());p['teacher_checkpoint']=str(Path(a.local_assets)/'checkpoints/self_forcing_dmd.pt');p['model_name']=os.path.relpath(Path(a.local_assets)/'wan_models/Wan2.1-T2V-1.3B',ROOT/'wan_models');return p
        backbone.protocol=local_protocol

def load_weights(a):
    return load(getattr(a,'model_seed',17),getattr(a,'checkpoint',CURRENT['checkpoint']),reader_checkpoint=a.reader_checkpoint or CURRENT['reader_checkpoint'],include_reader=getattr(a,'full_pset',True) and a.mode not in ('dense','dense_native'))

def make_runtime(engine,model,reader,phi,B,a):
    from .runtime import VideoRuntime
    from ..review_v1.ops import Config
    route=a.mode in ('route','tail','action','tensor','tensorfloat','compact','compact32','stream')
    tail=a.mode in ('tail','action','tensor','tensorfloat','compact','compact32','stream')
    action=a.mode in ('action','tensor','tensorfloat','compact','compact32','stream')
    tensor=a.mode in ('tensor','tensorfloat','compact','compact32','stream')
    rt=VideoRuntime(engine,model,reader,phi,B,full_pset=getattr(a,'full_pset',True) and a.mode!='dense',pset_steps='all' if getattr(a,'pset_policy',None) else getattr(a,'pset_steps','all'),route_m16=getattr(a,'route_m16',False),tile_batch=a.tile_batch,profile=getattr(a,'profile',False),validate=getattr(a,'validate',False),compile_route=route,compile_tail=tail,action_kernel=action,tensor_math=tensor,projection_float=a.mode=='tensorfloat',compact_reader=a.mode in ('compact','compact32','stream'),stream_attention=a.mode=='stream',stream_tail_mode=a.stream_tail_mode,stream_warps=a.stream_warps,bf16_qk=a.bf16_qk,split_summary=a.split_summary,fused_risk=a.fused_risk,static_route=a.static_route or bool(getattr(a,'pset_policy',None)),config=Config(history_pages=a.history_pages),legacy_components=a.legacy_components,projected_risk=a.projected_risk,cached_logmass=a.cached_logmass,coeff_bf16=getattr(a,'coeff_bf16',False))
    rt.batch_graph.native_route_qk = (getattr(a, 'native_route_qk', False) and rt.stream_attention
                                      and rt.route_m16 and rt.batch_graph.probes_only and a.bf16_qk)
    rt.batch_graph.scale_in_qk = rt.batch_graph.native_route_qk and getattr(a, 'scale_in_qk', False)
    if getattr(a,'grouping_dataflow',False) or getattr(a,'grouping_softmax',False) or getattr(a,'grouping_spatial',False):
        from .grouping_dataflow import GraphGrouping
        rt.group_graph=GraphGrouping(rt.config,contiguous_softmax=getattr(a,'grouping_softmax',False),spatial_softmax=getattr(a,'grouping_spatial',False),paired_gather=getattr(a,'group_gathers',False),one_hot=getattr(a,'group_gathers',False))
    if getattr(a,'selected_tie_order',False) and rt.legacy_attention and a.mode=='stream':
        rt.batch_graph.reuse_selected_ids='ties'
    rt.direct_output=getattr(a,'direct_output',False) and rt.legacy_attention
    for store in rt.stores.values():store.contiguous_commit=getattr(a,'contiguous_commit',False)
    width=getattr(a,'clean_stream_width',0)
    if width and a.mode=='stream' and getattr(a,'group_gathers',False) and getattr(a,'grouping_spatial',False) and not getattr(a,'block_graph',False):
        from .group_batch import ParallelGrouping
        rt.clean_group_batch=width
        exact=getattr(a,'exact_group_direction',False)
        features=getattr(a,'group_feature_bundle',False)
        rt.batched_group=ParallelGrouping(rt.config,width,exact_small=exact,fused_norm=exact,
                                          fused_features=features,staged_softmax=features)
        if getattr(a,'group_radix',False):
            for group in rt.batched_group.groupers:group.fused_sort='radix'
        import torch
        rt.commit_stream=torch.cuda.Stream(device=rt.counters.device)
    rt.resident_route=getattr(a,'resident_route',False)
    rt.cached_projection=getattr(a,'cached_projection',False)
    rt.workspace.aligned_coefficients=getattr(a,'aligned_coefficients',False)
    rt.workspace.page_prefetch=getattr(a,'page_prefetch',False)
    rt.workspace.column_mass=getattr(a,'column_mass',False)
    rt.workspace.anchor_scale=getattr(a,'anchor_scale',False)
    rt.workspace.fused_exact_tail=getattr(a,'fused_exact_tail',False)
    rt.workspace.tail_merge=getattr(a,'tail_merge',False)
    rt.workspace.strided_tail=getattr(a,'strided_tail',False)
    if getattr(a,'pset_policy',None):
        import json
        import torch
        torch._dynamo.config.cache_size_limit=64
        torch._dynamo.config.accumulated_cache_size_limit=512
        policy=json.loads(Path(a.pset_policy).read_text())
        if 'schedule' in policy:rt.set_pset_schedule(policy['schedule'])
        else:rt.set_pset_scope(policy.get('layers'),policy.get('heads'),policy.get('steps'))
    if getattr(a,'inline_risk_projection',False) and a.projected_risk:
        from .risk_projected import risk_inline
        rt.batch_graph.risk_function=risk_inline
        rt.batch_graph.projected_route_inputs=(getattr(a,'projected_route_inputs',False) and
            rt.batch_graph.native_route_qk and not getattr(a,'risk_online',False) and
            not getattr(a,'capture_route',False) and not getattr(a,'block_graph',False))
        if (getattr(a,'page_route',False) and rt.batch_graph.projected_route_inputs and
            rt.batch_graph.reuse_selected_ids=='ties' and a.cached_logmass):
            rt.batch_graph.direct_risk_bound=getattr(a,'direct_risk_bound',False)
            rt.batch_graph.matched_probabilities=True
            rt.batch_graph.fixed_page_sets=True
            if rt.batch_graph.direct_risk_bound and getattr(a,'rechecked_mass',False):
                rt.batch_graph.partial_mass='recheck'
                rt.batch_graph.radix_select=getattr(a,'radix_select',False)
    if getattr(a,'risk_online',False):
        assert a.projected_risk
        from .risk_online import risk
        rt.batch_graph.risk_function=risk
    if getattr(a,'summary_moments',CURRENT['summary_moments']):
        assert a.split_summary
        from .summary_moments import build
        rt.summary_window=getattr(a,'summary_window',False)
        rt.summary_graph=partial(build,cuda_pool=getattr(a,'cuda_pool',CURRENT['cuda_pool']))
    from .backbone_ops import install
    for name,patch in install(engine,a).items():setattr(rt,name,patch)
    return rt
