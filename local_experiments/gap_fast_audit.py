import gc,json,time,torch
from pathlib import Path
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .gap_runtime import GapRuntime
from .gap_fast_runtime import GapFastRuntime as GapGraphRuntime
from .learned_cache.io import atomic,digest
ROOT=Path('artifacts/four_gaps_20261031')
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(713);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();emb=torch.load('artifacts/broad_coverage_20261030/embeddings/city_walk.pt',weights_only=True).cuda();reference=None;weights=None;rows=[]
    for label,cls in [('reference',GapRuntime),('graph',GapGraphRuntime)]:
        rt=cls(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        if weights is None:weights={k:v.detach().cpu().clone() for k,v in rt.model.state_dict().items()}
        else:rt.model.load_state_dict(weights)
        try:
            start=time.perf_counter();zs=[generate_chunk(engine,emb,0,c).cpu() for c in range(9)];torch.cuda.synchronize()
            if reference is None:reference=zs
            else:assert all(torch.equal(a,b) for a,b in zip(reference,zs)),'Graph route changed actual rollout'
            rows.append(dict(variant=label,seconds_including_cold_capture=time.perf_counter()-start,runtime=rt.finish()));print(label,'exact',flush=True)
        finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
    atomic(ROOT/'fast_checks.json',dict(passed=True,rows=rows,code={n:digest(Path('local_experiments')/n) for n in ['gap_fast_runtime.py','gap_graph_runtime.py','gap_runtime.py','gap_route.py','gap_model.py']}))
if __name__=='__main__':main()
