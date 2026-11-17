"""Shared accepted deployment configuration for benchmark and resident workers."""
from types import SimpleNamespace
from pathlib import Path
import os
from .weights import ROOT,load

CURRENT=dict(mode='stream',model_seed=17,checkpoint=str(ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt'),reader_checkpoint=str(ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt'),tile_batch=72,history_pages=360,stream_tail_mode=4,stream_warps=8,bf16_qk=True,split_summary=True,fused_risk=False,static_route=False,moment_scores=True,late_reader=True,blocked_scores=True,packed_scores=True,reader_page32=False,lazy_scores=False,reader_split=True,reader_layout=True,reader_layout_kind='a',scan_exclude=True,projected_risk=True,cached_logmass=True,legacy_components='rope,attention',profile=False,validate=False,local_assets=None,reader_output32=True,coeff_bf16=True,endpoint_den=True,cached_endpoints=True)

def current(**overrides):return SimpleNamespace(**(CURRENT|overrides))
def identity(a):return {k:getattr(a,k,v) for k,v in CURRENT.items() if k!='local_assets'}
def configure(a):
    from . import scores,reader_reduce,scores_moments,pipeline_graph,runtime
    scores.SCAN_EXCLUDE=a.scan_exclude
    reader_reduce.PAGE32=a.reader_page32;reader_reduce.SPLIT=a.reader_split;reader_reduce.LAYOUT=a.reader_layout or a.reader_split;reader_reduce.LAYOUT_KIND=a.reader_layout_kind
    reader_reduce.WARPS=4;reader_reduce.SHARD=128;reader_reduce.AGG32=True;reader_reduce.PACKED=a.packed_scores;reader_reduce.MOMENTS=a.moment_scores
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
    return load(getattr(a,'model_seed',17),getattr(a,'checkpoint',CURRENT['checkpoint']),reader_checkpoint=a.reader_checkpoint or CURRENT['reader_checkpoint'])

def make_runtime(engine,model,reader,phi,B,a):
    from .runtime import VideoRuntime
    from ..review_v1.ops import Config
    route=a.mode in ('route','tail','action','tensor','tensorfloat','compact','compact32','stream')
    tail=a.mode in ('tail','action','tensor','tensorfloat','compact','compact32','stream')
    action=a.mode in ('action','tensor','tensorfloat','compact','compact32','stream')
    tensor=a.mode in ('tensor','tensorfloat','compact','compact32','stream')
    return VideoRuntime(engine,model,reader,phi,B,full_pset=a.mode!='dense',tile_batch=a.tile_batch,profile=getattr(a,'profile',False),validate=getattr(a,'validate',False),compile_route=route,compile_tail=tail,action_kernel=action,tensor_math=tensor,projection_float=a.mode=='tensorfloat',compact_reader=a.mode in ('compact','compact32','stream'),stream_attention=a.mode=='stream',stream_tail_mode=a.stream_tail_mode,stream_warps=a.stream_warps,bf16_qk=a.bf16_qk,split_summary=a.split_summary,fused_risk=a.fused_risk,static_route=a.static_route,config=Config(history_pages=a.history_pages),legacy_components=a.legacy_components,projected_risk=a.projected_risk,cached_logmass=a.cached_logmass,coeff_bf16=getattr(a,'coeff_bf16',False))
