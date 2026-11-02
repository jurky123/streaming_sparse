"""Real-layer routing weight and normalizer experiments; no stored QKV."""
import torch,json,triton as tr
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_tail_packed_runtime import GapTailPackedRuntime
from .gap_route_input_kernels import query
from .gap_route_bf16qk import project
from .gap_scaled_stats import scaled_stats as ref_stats
from .gap_scaled_stats_fast import scaled_stats as fast_stats
from .gap_weight_ordered import weight as ref_weight
from .gap_weight_linear import weight
from .gap_probability_stats import probability_stats
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard,RunLock
@torch.no_grad()
def main():
    root=Path('artifacts/gap_route_stats_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;disk_guard(root,50,1)
        engine=Engine();rt=GapTailPackedRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model']);old=rt.decision;rows=[];seen=set()
        def decision(l,q,pk,*args):
            if l in [1,14,29] and rt.n[l]==338 and l not in seen:
                seen.add(l);qp=query(q,rt.probe_offsets);h,t,n,d=qp.shape;raw=torch.empty(h,t*n,pk.shape[1]*4,device=q.device);project(qp.reshape(h,t*n,d),pk.flatten(1,2),raw,bm=64,bn=64,bk=128,mode=2);logits=raw.view(h,t,n,pk.shape[1],4)
                z,w=ref_stats(logits,order=1,div=1,exp=0);prob=z.softmax(-1);mass,*_=probability_stats(prob);ref=ref_weight(prob,w,mass,block=64)
                for b in [128,256,512,1024]:
                    out=weight(prob,w,mass,block=b);row=dict(layer=l,kind='weight',block=b,exact=torch.equal(out,ref),max_error=float((out-ref).abs().max()),ms=tr.testing.do_bench_cudagraph(lambda:weight(prob,w,mass,block=b),rep=100),reference_ms=tr.testing.do_bench_cudagraph(lambda:ref_weight(prob,w,mass,block=64),rep=100));rows.append(row);print(json.dumps(row),flush=True)
                for b in [128,256,512]:
                    zz,ww=fast_stats(logits,order=1,div=1,block=b);row=dict(layer=l,kind='fast_stats',block=b,z_error=float((zz-z).abs().max()),w_error=float((ww-w).abs().max()),ms=tr.testing.do_bench_cudagraph(lambda:fast_stats(logits,order=1,div=1,block=b),rep=100),reference_ms=tr.testing.do_bench_cudagraph(lambda:ref_stats(logits,order=1,div=1,exp=0),rep=100));rows.append(row);print(json.dumps(row),flush=True)
                atomic(root/'checks.json',dict(status='running',rows=rows))
            return old(l,q,pk,*args)
        rt.decision=decision;emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(seen)==3 and not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
