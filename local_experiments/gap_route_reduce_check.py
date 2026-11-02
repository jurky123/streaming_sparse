"""Fused route feature-score evaluation on real inputs from three layers."""
import torch,triton as tr,json,math
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_score_runtime import GapRouteScoreRuntime as GapGroupFusedRuntime
from .gap_route_score_scalar_ops import postprocess as reference
from .gap_route_reduce_ops import postprocess as candidate
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,RunLock,disk_guard
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    root=Path('artifacts/gap_route_reduce_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapGroupFusedRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model']);old=rt.decision;seen=set();rows=[]
        def decision(l,q,pk,base,detail,cov,vvar):
            result=old(l,q,pk,base,detail,cov,vvar)
            if l in [1,14,29] and rt.n[l]==338 and l not in seen:
                seen.add(l);h,p,_,d=pk.shape;t=q.shape[1]//64;n=rt.probe_count
                args=(rt.graw[:h*t*n*p*4].view(h,t,n,p,4),rt.gqp,rt.gbase[:base.numel()].view_as(base),rt.gdetail[:detail.numel()].view_as(detail),rt.gcov[:cov.numel()].view_as(cov),rt.gvar[:vvar.numel()].view_as(vvar),rt.gkvar[:h*p].view(h,p),rt.gw,rt.ggate,rt.ghaar)
                a=reference(*args)[0];b=candidate(*args)[0]
                row=dict(layer=l,selected_exact=torch.equal(a,b),changed_tiles=int((a.sort(-1).values!=b.sort(-1).values).any(-1).sum()),native_ms=tr.testing.do_bench_cudagraph(lambda:reference(*args),rep=150),candidate_ms=tr.testing.do_bench_cudagraph(lambda:candidate(*args),rep=150))
                rows.append(row);print(json.dumps(row),flush=True);atomic(root/'checks.json',dict(status='running',rows=rows))
            return result
        rt.decision=decision;emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(rows)==3 and not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,scope='Real-layer route postprocessing; no video acceptance claim.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
