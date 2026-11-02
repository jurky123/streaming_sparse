"""Real layer route projection fidelity and speed, no cached QKV dumps."""
import torch,json,triton as tr
from .broad_common import Engine,ROOT as DATA
from .gap_tail_packed_runtime import GapTailPackedRuntime as GapDirectSummaryRuntime
from .gap_route_input_kernels import query
from .gap_route_bf16qk import project
from .gap_route_bf16_stats import project_stats
from .gap_scaled_stats_fast import scaled_stats
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard
from pathlib import Path
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/gap_route_stats_20261101/fused');root.mkdir(parents=True,exist_ok=True);disk_guard(root,50,1)
    engine=Engine();rt=GapDirectSummaryRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
    rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
    old=rt.decision;rows=[];seen=set()
    def decision(l,q,pk,*args):
        if l in [1,14,29] and rt.n[l]==338 and l not in seen:
            seen.add(l);qp=query(q,rt.probe_offsets);h,t,n,d=qp.shape;qq=qp.reshape(h,t*n,d);kk=pk.flatten(1,2)
            raw=torch.empty(h,t*n,kk.shape[1],device=q.device)
            def separate():
                project(qq,kk,raw,bm=64,bn=64,bk=128,mode=2)
                return scaled_stats(raw.view(h,t,n,-1,4))
            zr,wr=separate();z=torch.empty_like(zr);w=torch.empty_like(wr)
            compiled=project_stats(qq,kk,z,w)
            row=dict(layer=l,z_exact=torch.equal(z,zr),w_exact=torch.equal(w,wr),z_max=float((z-zr).abs().max()),w_max=float((w-wr).abs().max()),registers=compiled.n_regs,spills=compiled.n_spills,separate_ms=tr.testing.do_bench_cudagraph(separate,rep=150),fused_ms=tr.testing.do_bench_cudagraph(lambda:project_stats(qq,kk,z,w),rep=150))
            rows.append(row);print(json.dumps(row),flush=True);atomic(root/'checks.json',dict(rows=rows,status='running'))

        return old(l,q,pk,*args)
    rt.decision=decision
    try:
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(rows)==3 and not device_peers();atomic(root/'checks.json',dict(rows=rows,status='complete',scope='Real-layer projection+statistics fusion relative to BF16 high/low route and fast normalizers. Not full-video acceptance.'))
    finally:rt.close()
if __name__=='__main__':main()
