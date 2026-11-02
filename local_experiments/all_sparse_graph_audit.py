"""Actual rollout equality, module CUDA timings and workspace audit."""
import gc,json,time,torch
from pathlib import Path
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .all_sparse_runtime import AllSparseRuntime
from .all_sparse_graph_inference import GraphInferenceRuntime as InferenceRuntime
from .learned_cache.io import atomic,digest,RunLock
ROOT=Path('artifacts/sparse_expansion_20261031/graph_inference_audit');TRAIN=ROOT.parent/'all_sparse_train'
@torch.no_grad()
def main():
    lock=RunLock(ROOT)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(713)
        engine=Engine();emb=torch.load('artifacts/broad_coverage_20261030/embeddings/city_walk.pt',weights_only=True).cuda();rows=[]
        for label in ['initial','epoch1']:
            expected=None
            for name,cls in [('reference',AllSparseRuntime),('zero_copy',InferenceRuntime)]:
                atomic(ROOT/'status.json',dict(status='running',checkpoint=label,variant=name))
                rt=cls(engine,'artifacts/full_route_grad_20261031/epoch1.pt');rt.model.load_state_dict(torch.load(TRAIN/(label+'.pt'),map_location='cpu',weights_only=False)['model']);events=[]
                def timed(label,fn):
                    def wrapped(*a,**kw):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();result=fn(*a,**kw);end.record();events.append((label,start,end));return result
                    return wrapped
                rt.commit=timed('commit_inclusive',rt.commit);rt.decision=timed('route',rt.decision)
                for m in rt.model.layers.values():m.pool=timed('pool_inclusive',m.pool);m.group_features=timed('group_features',m.group_features)
                import local_experiments.all_sparse_attention as att
                import local_experiments.all_sparse_runtime as refmod
                import local_experiments.all_sparse_inference as optmod
                oldcomp=att.compensate;oldnative=att.native
                att.compensate=timed('compensation',oldcomp);target=refmod if name=='reference' else optmod;target.native=timed('exact_plus_compensation',oldnative)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                try:
                    latents=[generate_chunk(engine,emb,0,c).cpu() for c in range(9)];torch.cuda.synchronize();wall=time.perf_counter()-start
                    if expected is None:expected=latents
                    else:assert all(torch.equal(a,b) for a,b in zip(expected,latents)),'Optimized rollout changed'
                    profile={}
                    for key,a,b in events:
                        d=profile.setdefault(key,dict(calls=0,total_ms=0.));d['calls']+=1;d['total_ms']+=a.elapsed_time(b)
                    for d in profile.values():d['mean_ms']=d['total_ms']/d['calls']
                    row=dict(checkpoint=label,variant=name,profile=profile,profiled_9chunk_wall_seconds=wall,peak_allocated_bytes=torch.cuda.max_memory_allocated(),latent_exact=True)
                    rows.append(row);atomic(ROOT/f'{label}_{name}.json',row);print(json.dumps(row),flush=True)
                finally:att.compensate=oldcomp;target.native=oldnative;rt.close();del rt;gc.collect();torch.cuda.empty_cache()
        atomic(ROOT/'result.json',dict(passed=True,rows=rows,code={n:digest(Path('local_experiments')/n) for n in ['all_sparse_graph_inference.py','all_sparse_inference.py','all_sparse_runtime.py']},scope='9 chunks, seed0 city_walk, initial/epoch1 exact equality; CUDA events include nested timings; no VAE, warm-up not excluded, not E2E speed claim'))
        atomic(ROOT/'status.json',dict(status='complete'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
