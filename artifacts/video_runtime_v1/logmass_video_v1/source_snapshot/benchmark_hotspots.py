"""Single-video Full_Pset end-to-end A/B with full-shape warmup and stage profile."""
import argparse,gc,time,json,statistics,os
from pathlib import Path
import torch
from .weights import load,ROOT
from .runtime import VideoRuntime
from ..review_v1.ops import Config
from . import pipeline_graph,runtime,scores,attention
from ..broad_common import Engine
from ..rollout_suite import generate_chunk
from ..learned_cache.io import atomic,digest

@torch.no_grad()
def run(a):
    scores.SCAN_EXCLUDE=a.scan_exclude
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise ValueError('Use fresh output')
    source={};snapshot=out/'source';snapshot.mkdir()
    import hashlib
    for p in Path(__file__).parent.glob('*.py'):
        data=p.read_bytes();source[p.name]=hashlib.sha256(data).hexdigest();(snapshot/p.name).write_bytes(data)
    model,reader,phi,B,identity=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt',reader_checkpoint=getattr(a,'reader_checkpoint',None))
    if a.mode in ('compact32','stream'):
        from . import reader_reduce
        reader_reduce.PAGE32=a.reader_page32;reader_reduce.SPLIT=a.reader_split;reader_reduce.LAYOUT=a.reader_layout or a.reader_split;reader_reduce.LAYOUT_KIND=a.reader_layout_kind;reader_reduce.WARPS=4;reader_reduce.SHARD=128;reader_reduce.AGG32=True;reader_reduce.PACKED=a.packed_scores;reader_reduce.MOMENTS=a.moment_scores
    scorer=scores.grouped_scores if a.mode=='baseline' else scores.fused_grouped_scores
    pipeline_graph.grouped_scores=scorer;runtime.grouped_scores=scorer
    if a.local_assets:
        from streaming_sparse.v08 import backbone
        original_protocol=backbone.protocol
        def local_protocol():
            p=dict(original_protocol());p['teacher_checkpoint']=str(Path(a.local_assets)/'checkpoints/self_forcing_dmd.pt');p['model_name']=os.path.relpath(Path(a.local_assets)/'wan_models/Wan2.1-T2V-1.3B',ROOT/'wan_models');return p
        backbone.protocol=local_protocol
    engine=Engine();engine.wrapper.requires_grad_(False)
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    from . import scores_moments
    scores_moments.BLOCKED=a.blocked_scores;scores_moments.LAZY=a.lazy_scores
    if a.late_reader:
        from . import reader_reduce
        from .reader_late import reader_kernel
        reader_reduce.reader_kernel=reader_kernel
    rt=None if a.mode=="dense_native" else VideoRuntime(engine,model,reader,phi,B,full_pset=a.mode!="dense",tile_batch=a.tile_batch,compile_route=a.mode in ("route","tail","action","tensor","tensorfloat","compact","compact32","stream"),compile_tail=a.mode in ("tail","action","tensor","tensorfloat","compact","compact32","stream"),action_kernel=a.mode in ("action","tensor","tensorfloat","compact","compact32","stream"),tensor_math=a.mode in ("tensor","tensorfloat","compact","compact32","stream"),projection_float=a.mode=="tensorfloat",compact_reader=a.mode in ("compact","compact32","stream"),stream_attention=a.mode=="stream",stream_tail_mode=a.stream_tail_mode,stream_warps=a.stream_warps,bf16_qk=a.bf16_qk,split_summary=a.split_summary,fused_risk=a.fused_risk,static_route=a.static_route,config=Config(history_pages=a.history_pages),legacy_components=a.legacy_components,projected_risk=a.projected_risk,cached_logmass=a.cached_logmass)
    if rt is not None:rt.sparse=a.mode!="dense"
    bg=None
    if a.block_graph:
        if rt is None or a.mode=='dense':raise ValueError('Block capture requires sparse runtime')
        from .block_graph import BlockGraphs
        bg=BlockGraphs(engine,rt)
    def generate():
        engine.reset() if rt is None else rt.reset();zs=[]
        for chunk in range(a.chunks):zs.append(generate_chunk(engine,emb,701,chunk))
        return torch.cat(zs,1)
    try:
        start=time.perf_counter();z=generate();torch.cuda.synchronize();cold=time.perf_counter()-start
        rows=[]
        for rep in range(a.reps):
            torch.cuda.synchronize();start=time.perf_counter();z=generate();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            assert torch.isfinite(z).all();rows.append(elapsed);print(a.mode,rep,elapsed,flush=True)
        atomic(out/'latents.pt',z.cpu(),True)
        stages={}
        if rt is not None and not a.block_graph and not getattr(a,'skip_profile',False):
            rt.timings.enabled=True;generate();torch.cuda.synchronize();stages=rt.timings.finish(clear=True)
        if a.capture_route and rt is not None:
            key=max(rt.batch_graph.entries,key=lambda k:k[2]);entry=rt.batch_graph.entries[key]
            names=('q','theta','basis','own','offsets','haar','haar_float')
            saved={n:entry[n].cpu() for n in names}
            saved['summary']={n:getattr(entry['s'],n).cpu() for n in ('keys','mean','detail','stats')}
            saved['en']={n:v.cpu() for n,v in entry['en'].items()}
            atomic(out/'route_entry.pt',saved,True)
        result=dict(mode=a.mode,cached_logmass=a.cached_logmass,projected_risk=a.projected_risk,scan_exclude=a.scan_exclude,reader_page32=a.reader_page32,lazy_scores=a.lazy_scores,reader_split=a.reader_split,reader_layout=a.reader_layout,reader_layout_kind=a.reader_layout_kind,legacy_components=a.legacy_components,static_route=a.static_route,fused_risk=a.fused_risk,blocked_scores=a.blocked_scores,late_reader=a.late_reader,split_summary=a.split_summary,moment_scores=a.moment_scores,stream_tail_mode=a.stream_tail_mode,stream_warps=a.stream_warps,bf16_qk=a.bf16_qk,packed_scores=a.packed_scores,block_graph=a.block_graph,video='city_walk_s701',chunks=a.chunks,tile_batch=a.tile_batch,history_pages=a.history_pages,resolution=[512,768],fps=16,frames={9:105,21:240,41:480}[a.chunks],cold_seconds=cold,seconds=rows,median_seconds=statistics.median(rows),stages=stages,weights=identity,scope='generation only; full-length warmup excludes compile; separate instrumented stage run',source=source)
        atomic(out/'result.json',result);print(json.dumps(result),flush=True)
    finally:
        if bg is not None:bg.close()
        if rt is not None:rt.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','fused','tail','route','dense','dense_native','action','tensor','tensorfloat','compact','compact32','stream'],required=True);p.add_argument('--output',required=True);p.add_argument('--chunks',type=int,choices=[9,21],default=9);p.add_argument('--history-pages',type=int,choices=[338,360],default=360);p.add_argument('--stream-tail-mode',type=int,choices=[1,4],default=4);p.add_argument('--stream-warps',type=int,choices=[4,8],default=4);p.add_argument('--bf16-qk',action='store_true');p.add_argument('--split-summary',action='store_true');p.add_argument('--fused-risk',action='store_true');p.add_argument('--capture-route',action='store_true');p.add_argument('--static-route',action='store_true');p.add_argument('--moment-scores',action='store_true');p.add_argument('--late-reader',action='store_true');p.add_argument('--blocked-scores',action='store_true');p.add_argument('--packed-scores',action='store_true');p.add_argument('--block-graph',action='store_true');p.add_argument('--tile-batch',type=int,choices=[6,12,24,36,72],default=6);p.add_argument('--skip-profile',action='store_true');p.add_argument('--reader-checkpoint');p.add_argument('--legacy-components',default='attention');p.add_argument('--local-assets');p.add_argument('--cached-logmass',action='store_true');p.add_argument('--projected-risk',action='store_true');p.add_argument('--scan-exclude',action='store_true');p.add_argument('--reader-page32',action='store_true');p.add_argument('--lazy-scores',action='store_true');p.add_argument('--reader-split',action='store_true');p.add_argument('--reader-layout',action='store_true');p.add_argument('--reader-layout-kind',choices=['a','co','both'],default='a');p.add_argument('--reps',type=int,default=2);run(p.parse_args())
