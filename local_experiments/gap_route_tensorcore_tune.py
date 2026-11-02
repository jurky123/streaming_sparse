"""Real layer route projection fidelity and speed, no cached QKV dumps."""
import torch,json,triton as tr
from .broad_common import Engine,ROOT as DATA
from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .gap_route_input_kernels import query
from .gap_route_tensorcore import project
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard
from pathlib import Path
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/gap_broad_opt_20261031/route_tensorcore');root.mkdir(parents=True,exist_ok=True);disk_guard(root,50,1)
    engine=Engine();rt=GapDirectSummaryRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
    rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
    old=rt.decision;rows=[];seen=set()
    def decision(l,q,pk,*args):
        if l in [1,14,29] and rt.n[l]==338 and l not in seen:
            seen.add(l);qp=query(q,rt.probe_offsets);h,t,n,d=qp.shape;qq=qp.reshape(h,t*n,d);kk=pk.flatten(1,2)
            ref=torch.bmm(qq,kk.transpose(1,2));out=torch.empty_like(ref);project(qq,kk,out)
            for bm,bn,bk,mode in [(32,64,32,3),(64,64,128,3),(64,64,128,2),(32,64,128,2),(64,128,128,2)]:
                project(qq,kk,out,bm,bn,bk,mode)
                row=dict(layer=l,bm=bm,bn=bn,bk=bk,mode=mode,max_error=float((out-ref).abs().max()),nmse=float((out-ref).square().sum()/ref.square().sum()),native_ms=tr.testing.do_bench_cudagraph(lambda:torch.bmm(qq,kk.transpose(1,2),out=ref),rep=150),candidate_ms=tr.testing.do_bench_cudagraph(lambda:project(qq,kk,out,bm,bn,bk,mode),rep=150))
                rows.append(row);print(json.dumps(row),flush=True);atomic(root/'tuning.json',dict(rows=rows,status='running'))

        return old(l,q,pk,*args)
    rt.decision=decision
    try:
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(rows)==15 and not device_peers();atomic(root/'tuning.json',dict(rows=rows,status='complete',scope='Projection only; route decisions and video quality not yet validated.'))
    finally:rt.close()
if __name__=='__main__':main()
