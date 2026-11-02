"""Measure route dispatch overhead without profiler, retaining actual operands."""
import torch,json,time,triton as tr
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_half_runtime import GapRouteHalfRuntime as GapTailPackedRuntime
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard,RunLock
@torch.no_grad()
def main():
    root=Path('artifacts/gap_route_half_20261101/dispatch');lock=RunLock(root);rt=None
    try:
        assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;disk_guard(root,50,1)
        engine=Engine();rt=GapTailPackedRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model']);old=rt.decision;rows=[];seen=set()
        def decision(l,q,pk,*args):
            if l in [1,14,29] and rt.n[l]==338 and l not in seen:
                seen.add(l)
                def run():return old(l,q,pk,*args)
                for _ in range(5):run()
                ref=run()[0].clone()
                # This torch build disallows capturing CUDAGraph.replay. Flatten
                # the inner postprocess into the outer capture for the diagnostic.
                from types import SimpleNamespace
                from .gap_route_fused_stats_ops import postprocess
                h,p,_,d=pk.shape;t=q.shape[1]//64;n=rt.probe_count
                base,detail,cov,vvar=args
                graphargs=(rt.graw[:h*t*n*p*4].view(h,t,n,p,4),rt.gqp,rt.gbase[:base.numel()].view_as(base),rt.gdetail[:detail.numel()].view_as(detail),rt.gcov[:cov.numel()].view_as(cov),rt.gvar[:vvar.numel()].view_as(vvar),rt.gkvar[:h*p].view(h,p),rt.gw,rt.ggate,rt.ghaar,rt.route_logz[:h*t*n*p].view(h,t,n,p))
                original_graph=rt.graphs_new[p];original_output=rt.outputs_new[p]
                def flattened():rt.outputs_new[p]=postprocess(*graphargs)
                try:
                    rt.graphs_new[p]=SimpleNamespace(replay=flattened)
                    assert torch.equal(ref,run()[0])
                    graph_ms=tr.testing.do_bench_cudagraph(run,rep=150)
                finally:
                    rt.graphs_new[p]=original_graph;rt.outputs_new[p]=original_output
                assert torch.equal(ref,run()[0])
                trials=[]
                for _ in range(3):
                    torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();start=time.perf_counter()
                    for i in range(100):run()
                    host=(time.perf_counter()-start)*10;b.record();torch.cuda.synchronize();wall=(time.perf_counter()-start)*10
                    trials.append(dict(host_enqueue_ms=host,wall_ms=wall,cuda_event_including_queue_gaps_ms=a.elapsed_time(b)/100))
                row=dict(layer=l,outer_graph_ms=graph_ms,trials=trials);rows.append(row);atomic(root/'checks.json',dict(status='running',rows=rows));print(json.dumps(row),flush=True)
            return old(l,q,pk,*args)
        rt.decision=decision;emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(seen)==3 and not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,scope='Repeated fixed real-layer route inputs, warm; no profiler. Outer graph flattens the existing inner graph; includes complete route math but excludes replay/input staging cost and model-wide scheduling. Not a full-generation speedup.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
