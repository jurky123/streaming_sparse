"""Real layer route projection fidelity and speed, no cached QKV dumps."""
import torch,json,triton as tr
from .broad_common import Engine,ROOT as DATA
from .gap_tail_bf16qk_stage2_runtime import GapTailBF16QKStage2Runtime as GapDirectSummaryRuntime
from .gap_route_input_kernels import query
from .gap_route_packed import project
from .gap_route_bf16qk import project as reference_project
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard
from pathlib import Path
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/gap_route_packed_20261101');root.mkdir(parents=True,exist_ok=True);disk_guard(root,50,1)
    engine=Engine();rt=GapDirectSummaryRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
    rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
    old=rt.decision;rows=[];seen=set()
    def decision(l,q,pk,*args):
        if l in [1,14,29] and rt.n[l]==338 and l not in seen:
            seen.add(l);qp=query(q,rt.probe_offsets);h,t,n,d=qp.shape;qq=qp.reshape(h,t*n,d);kk=pk.flatten(1,2)
            ref=torch.empty(h,t*n,kk.shape[1],device=q.device);reference_project(qq,kk,ref,bm=64,bn=64,bk=128,mode=2);out=torch.empty_like(ref);project(qq,kk,out,bm=64,bn=64,bk=128,mode=2)
            row=dict(layer=l,exact=torch.equal(out,ref),max_error=float((out-ref).abs().max()),nmse=float((out-ref).square().sum()/ref.square().sum()),native_ms=tr.testing.do_bench_cudagraph(lambda:reference_project(qq,kk,ref,bm=64,bn=64,bk=128,mode=2),rep=150),packed_including_preprocessing_ms=tr.testing.do_bench_cudagraph(lambda:project(qq,kk,out,bm=64,bn=64,bk=128,mode=2),rep=150))
            rows.append(row);print(json.dumps(row),flush=True);atomic(root/'checks.json',dict(rows=rows,status='running'))
        return old(l,q,pk,*args)
    rt.decision=decision
    try:
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(rows)==3 and not device_peers();atomic(root/'checks.json',dict(rows=rows,status='complete',scope='Projection only; route decisions and video quality not yet validated.'))
    finally:rt.close()
if __name__=='__main__':main()
