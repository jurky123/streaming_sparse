"""Real Q/K/V/LUT exact sparse kernel resource and timing sweep; no dumps."""
import torch,triton as tr,json
from pathlib import Path
from triton.runtime.errors import OutOfResources
from .broad_common import Engine,ROOT as DATA
from .gap_route_combined_runtime import GapRouteCombinedRuntime
from .gap_exact_subquery_kernel import _attn_fwd as sub_kernel
from . import gap_flat_runtime as module
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,RunLock,disk_guard
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    root=Path('artifacts/gap_exact_tune_20261101');lock=RunLock(root);rt=None;original=module.flat_native
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapRouteCombinedRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model']);rows=[];seen=[0]
        def audit(q,k,v,pk,base,detail,basis,center,scale,lut,kernel):
            if pk.shape[1]==410 and seen[0]<3:
                sample=seen[0];seen[0]+=1;_,n,h,d=q.shape;ref=None
                for bm,warps,stages in [(64,4,3),(32,4,1),(32,4,2),(32,4,3),(32,2,1),(32,2,2),(16,2,1),(16,2,2)]:
                    out=torch.empty_like(q);lse=torch.empty(1,h,n,device=q.device)
                    def run():return (kernel if bm==64 else sub_kernel)[(n//bm,h)](q,k,v,d**-.5,lut.shape[-1],lut,lse,out,h,n,k.shape[1],n//64,d,bm,64,num_warps=warps,num_stages=stages)
                    row=dict(sample=sample,bm=bm,warps=warps,stages=stages)
                    try:
                        compiled=run();torch.cuda.synchronize()
                        if ref is None:ref=(out.clone(),lse.clone())
                        row.update(output_exact=torch.equal(out,ref[0]),lse_exact=torch.equal(lse,ref[1]),nmse=float((out.float()-ref[0].float()).square().sum()/ref[0].float().square().sum()),lse_max=float((lse-ref[1]).abs().max()),registers=compiled.n_regs,spills=compiled.n_spills,shared_bytes=compiled.metadata.shared,ms=tr.testing.do_bench_cudagraph(run,rep=150))
                    except OutOfResources as exc:row['unavailable']=str(exc)
                    rows.append(row);print(json.dumps(row),flush=True);atomic(root/'subquery_checks.json',dict(status='running',rows=rows))
            return original(q,k,v,pk,base,detail,basis,center,scale,lut,kernel)
        module.flat_native=audit;emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert seen[0]==3 and not device_peers();atomic(root/'subquery_checks.json',dict(status='complete',rows=rows,scope='Three real steady-state attention calls. Physical pages and routing tiles remain64, compute BQ may be16/32; no full-video claim.'))
    finally:
        module.flat_native=original
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
