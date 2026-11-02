"""Actual route decisions, complete-graph staging cost included in timings."""
import torch,json,time
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_qp_runtime import GapRouteQpRuntime as GapRouteCompleteGraphRuntime
from .gap_staged_history_runtime import GapStagedHistoryRuntime as GapPoolLinearRuntime
from .gap_route_query_project import gather_project
from .gap_route_input_kernels import query
from .gap_peer_watch import PeerWatch
import math,triton as tr
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard,RunLock
@torch.no_grad()
def main():
    root=Path('artifacts/gap_route_qp_20261102');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapRouteCompleteGraphRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16);rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model']);rows=[];seen=set();old=rt.decision
        def decision(l,q,pk,*args):
            if l in [1,14,29] and rt.n[l]==338 and l not in seen:
                seen.add(l)
                def before():return GapPoolLinearRuntime.decision(rt,l,q,pk,*args)
                def after():return old(l,q,pk,*args)
                ref=before()[0].clone();out=after()[0].clone()
                selected=query(q,rt.probe_offsets);basis=rt.model[l].basis
                expected=selected@basis.to(torch.bfloat16).float()[:,None]/math.sqrt(128)
                gathered,projected=gather_project(q,rt.probe_offsets,basis.contiguous())
                assert torch.equal(selected,gathered)
                projection_nmse=float((projected-expected).square().sum()/expected.square().sum())
                for _ in range(5):before();after()
                trials=[]
                for name,fn in [('before',before),('after',after),('after',after),('before',before)]:
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(100):fn()
                    torch.cuda.synchronize();trials.append(dict(variant=name,wall_ms=(time.perf_counter()-start)*10))
                row=dict(layer=l,exact=torch.equal(ref,out),lut_position_changed_fraction=float((ref!=out).float().mean()),projection_nmse=projection_nmse,trials=trials);rows.append(row);atomic(root/'checks.json',dict(status='running',rows=rows));print(json.dumps(row),flush=True)
            return old(l,q,pk,*args)
        rt.decision=decision;emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        with PeerWatch() as watch:
            for c in range(7):generate_chunk(engine,emb,101,c)
        atomic(root/'peer_samples.json',watch.summary());watch.validate()
        assert len(seen)==3 and not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,peer_monitor_validated=True,scope='Fixed real operands, baseline/candidate alternating; explicit Q/PK/basis staging included. Not full video.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
