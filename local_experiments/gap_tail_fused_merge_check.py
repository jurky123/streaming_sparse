"""Compare full compensation including transient BF16 packing, never dump QKV."""
import torch,triton as tr,json
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_pool_linear_runtime import GapPoolLinearRuntime as GapExactRoundRuntime
from .gap_tail_single import compensate as reference
from .gap_tail_fused_merge import compensate as candidate
from . import gap_flat_runtime as module
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,RunLock,disk_guard
from .gap_group_isolated import device_peers
from .gap_peer_watch import PeerWatch
@torch.no_grad()
def main():
    root=Path('artifacts/gap_tail_fused_merge_20261101');lock=RunLock(root);rt=None;original=module.flat_native
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapExactRoundRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model']);rows=[]
        def audit(q,k,v,pk,base,detail,basis,center,scale,lut,kernel):
            if pk.shape[1]==410 and len(rows)<3:
                _,n,h,d=q.shape;exact=torch.empty_like(q);lse=torch.empty(1,h,n,device=q.device)
                kernel[(n//64,h)](q,k,v,d**-.5,lut.shape[-1],lut,lse,exact,h,n,k.shape[1],n//64,d,64,64,num_warps=4,num_stages=3)
                gamma=torch.ones(h,device=q.device);out=exact.clone()
                def run(fn):
                    out.copy_(exact)
                    return fn(q,pk,base,detail,lut,lut,out,lse,basis,center,gamma,scale,mode=1)
                with PeerWatch() as watch:
                    ref=tuple(x.clone() for x in run(reference));got=run(candidate)
                    row=dict(sample=len(rows),exact=[torch.equal(a,b) for a,b in zip(ref,got)],nmse=[float((a.float()-b.float()).square().sum()/a.float().square().sum()) for a,b in zip(ref,got)],reference_ms=tr.testing.do_bench_cudagraph(lambda:run(reference),rep=150),candidate_ms=tr.testing.do_bench_cudagraph(lambda:run(candidate),rep=150))
                atomic(root/f'peer_samples_{len(rows)}.json',watch.summary());watch.validate()
                rows.append(row);atomic(root/'checks.json',dict(status='running',rows=rows));print(json.dumps(row),flush=True)
            return original(q,k,v,pk,base,detail,basis,center,scale,lut,kernel)
        module.flat_native=audit;emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(rows)==3 and not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,peer_monitor_validated=True,scope='Includes transient packing, allocation, decode, mask, attention, merge and identical output reset. No persistent cache increase; not E2E validation.'))
    finally:
        module.flat_native=original
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
