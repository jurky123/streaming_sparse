"""Actual every-layer LUT/summary equality, initial and trained checkpoints."""
import json,time
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .route_graph_runtime import GraphRuntime,RouteGraph
from .route_norm_runtime import NormRuntime as FusedRuntime
from .learned_cache.io import atomic,disk_guard,RunLock
from .improve_accept import Telemetry
ROOT=Path('artifacts/route_kernel_20261031/norm')
@torch.no_grad()
def main():
    lock=RunLock(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Engine();engine.shared_route_graph=RouteGraph();rows=[];times=[];tele=Telemetry();tele.start()
    try:
        for model,cp in [('initial','artifacts/high_sparse_gain_20261031/model.pt'),('trained','artifacts/full_route_grad_20261031/epoch1.pt')]:
            for case in ['basketball_court','butterfly_garden']:
                rt=FusedRuntime(engine,cp);base=rt.route
                def route(l,q,p):
                    fast_router=rt.router;rt.router=engine.shared_route_graph
                    old=GraphRuntime.route(rt,l,q,p).clone();pk=rt.active_pk.clone();code=rt.active_codes.clone()
                    rt.router=fast_router;new=base(l,q,p)
                    diff=int((old!=new).any(-1).sum());pkdiff=int((pk!=rt.active_pk).sum());cdiff=int((code!=rt.active_codes).sum())
                    row=dict(model=model,case=case,layer=l,chunk=engine.current['chunk'],step=engine.current['step'],history_pages=p,different_tiles=diff,pk_diff=pkdiff,code_diff=cdiff)
                    if diff or pkdiff or cdiff:atomic(ROOT/'failure.json',row);raise AssertionError(row)
                    rows.append(row)
                    if model=='trained' and case=='basketball_court' and l==14 and engine.current['chunk']==5 and engine.current['step']==0:
                        for rep in range(6):
                            for method in (['graph','fused'] if rep%2==0 else ['fused','graph']):
                                rt.router=engine.shared_route_graph if method=='graph' else fast_router
                                fn=(lambda:GraphRuntime.route(rt,l,q,p)) if method=='graph' else (lambda:base(l,q,p))
                                for _ in range(3):fn()
                                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();a.record();start=time.perf_counter()
                                for _ in range(20):fn()
                                b.record();b.synchronize();times.append(dict(rep=rep,method=method,wall_ms=(time.perf_counter()-start)*50,gpu_ms=a.elapsed_time(b)/20))
                        rt.router=fast_router;new=base(l,q,p)
                    return new
                rt.route=route
                try:
                    disk_guard(ROOT,50,1);emb=torch.load(f'artifacts/mass_variance_20261030/embeddings/{case}.pt',weights_only=True).cuda()
                    latents=torch.cat([generate_chunk(engine,emb,1001,c).cpu() for c in range(9)],1)
                    previous=Path('artifacts/full_route_grad_20261031/acceptance/early')/case/(model+'.latents.pt')
                    assert torch.equal(latents,torch.load(previous,weights_only=True)),(model,case,'prefix differs')
                    atomic(ROOT/'check_progress.json',dict(model=model,case=case,events=len(rows),prefix_bit_exact=True));print(model,case,len(rows),'passed',flush=True)
                finally:rt.close()
        load=tele.finish();atomic(ROOT/'checks.json',dict(passed=True,events=len(rows),rows=rows,timings=times,telemetry=load));print(json.dumps(times),flush=True)
    finally:
        if not tele.stop.is_set():tele.finish()
        lock.close()
if __name__=='__main__':main()
