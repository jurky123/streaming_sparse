"""Gather microbenchmarks on actual history-commit inputs; no tensor dumps."""
import json,torch,triton as tr
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_qp_runtime import GapRouteQpRuntime
from .rollout_suite import generate_chunk
from .gap_peer_watch import PeerWatch
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard,RunLock
from .gap_reorder_kernel import reorder
from . import gap_commit_graph_runtime as mod
ROOT=Path('artifacts/gap_reorder_20261102')
@torch.no_grad()
def main():
    lock=RunLock(ROOT);rt=None;original=mod.reorder;samples={}
    try:
        assert not device_peers();disk_guard(ROOT,50,1);torch.set_num_threads(4)
        engine=Engine();rt=GapRouteQpRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        def collect(x,order):
            key=(tuple(x.shape),str(x.dtype),tuple(x.stride()))
            if key not in samples and not torch.cuda.is_current_stream_capturing():
                samples[key]=(x.clone(memory_format=torch.preserve_format),order.clone())
            return original(x,order)
        mod.reorder=collect
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        with PeerWatch() as watch:
            generate_chunk(engine,emb,101,0)
            mod.reorder=original
            rows=[]
            for key,(x,order) in samples.items():
                ref=original(x,order)
                baseline=tr.testing.do_bench_cudagraph(lambda:original(x,order),rep=100)*1000
                configs=[]
                for b in [4,8,16,32]:
                    got=reorder(x,order,b);assert torch.equal(ref,got)
                    us=tr.testing.do_bench_cudagraph(lambda b=b:reorder(x,order,b),rep=100)*1000
                    configs.append(dict(block=b,microseconds=us,exact=True))
                row=dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),baseline_us=baseline,candidates=configs)
                rows.append(row);print(json.dumps(row),flush=True)
        atomic(ROOT/'peer_samples.json',watch.summary());watch.validate();assert not device_peers() and len(rows)>=3
        atomic(ROOT/'checks.json',dict(status='complete',peer_monitor_validated=True,rows=rows,scope='Actual history commit gather inputs; isolated captured GPU microbenchmark, not full commit or full video.'))
    finally:
        mod.reorder=original
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
