"""Real-QKV LUT identity and warmed whole-route paired microtimings."""
import json,time
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .high_sparse_gain_runtime import GainRuntime
from .route_graph_runtime import GraphRuntime
from .learned_cache.io import atomic
ROOT=Path('artifacts/route_graph_20261031')
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Engine();rt=GraphRuntime(engine,'artifacts/high_sparse_gain_20261031/model.pt',1.);base=rt.route;rows=[];times=[]
    def route(l,q,p):
        old=GainRuntime.route(rt,l,q,p).clone();new=base(l,q,p)
        difference=int((old!=new).any(-1).sum());assert difference==0,(engine.current,difference)
        rows.append(dict(layer=l,chunk=engine.current['chunk'],step=engine.current['step'],history_pages=p,different_tiles=difference))
        if l==14 and engine.current['chunk']==5 and engine.current['step']==0:
            for rep in range(6):
                for label,fn in [('eager',lambda:GainRuntime.route(rt,l,q,p)),('graph',lambda:base(l,q,p))]:
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(10):fn()
                    torch.cuda.synchronize();times.append(dict(rep=rep,method=label,ms=(time.perf_counter()-start)*100))
            new=base(l,q,p)
        return new
    rt.route=route
    try:
        emb=torch.load('artifacts/mass_variance_20261030/embeddings/basketball_court.pt',weights_only=True).cuda()
        for c in range(9):generate_chunk(engine,emb,1001,c)
        assert len(rows)==812
        result=dict(passed=True,events=len(rows),rows=rows,timings=times,runtime=rt.finish(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic(ROOT/'checks.json',result);print(json.dumps(dict(passed=True,events=len(rows),timings=times,persistent_bytes=rt.persistent_bytes)),flush=True)
    finally:rt.close()
if __name__=='__main__':main()
