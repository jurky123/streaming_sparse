"""Actual route operands: FP16 within workspace and fused kvar experiments."""
import torch,json,triton as tr
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_fused_tuned_runtime import GapRouteFusedTunedRuntime,project_stats
from .gap_route_input_kernels import query,kvar
from .gap_kvar_fused import kvar as fused_kvar
from .gap_route_fused_stats_ops import postprocess
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,disk_guard,RunLock
@torch.no_grad()
def main():
    root=Path('artifacts/gap_route_half_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;disk_guard(root,50,1)
        engine=Engine();rt=GapRouteFusedTunedRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model']);old=rt.decision;rows=[];seen=set()
        def decision(l,q,pk,base,detail,cov,vvar):
            if l in [1,14,29] and rt.n[l]==338 and l not in seen:
                seen.add(l);qp=query(q,rt.probe_offsets);h,t,n,d=qp.shape;p=pk.shape[1];qq=qp.reshape(h,t*n,d);kk=pk.flatten(1,2)
                z=torch.empty(h,t,n,p,device=q.device);wh=torch.empty(h,t,n,p,4,device=q.device,dtype=torch.float16);wf=torch.empty_like(wh,dtype=torch.float32)
                project_stats(qq,kk,z,wf);zref=z.clone();project_stats(qq,kk,z,wh)
                m=rt.model[l];proj=qp@m.basis.to(torch.bfloat16).float()[:,None]/d**.5;kv=kvar(pk);fk=fused_kvar(pk)
                def run(w,variance):return postprocess(w,proj,base,detail,cov,vvar,variance,m.route,m.residual_gate,rt.ghaar,z)[0]
                reference=run(wf,kv).clone();half=run(wh,kv).clone();fused=run(wf,fk).clone()
                row=dict(layer=l,z_exact=torch.equal(z,zref),within_max_error=float((wf-wh.float()).abs().max()),half_changed_ids=int((reference!=half).sum()),kvar_changed_ids=int((reference!=fused).sum()),ids_total=reference.numel(),kvar_max_error=float((kv-fk).abs().max()),projection_float_ms=tr.testing.do_bench_cudagraph(lambda:project_stats(qq,kk,z,wf),rep=100),projection_half_ms=tr.testing.do_bench_cudagraph(lambda:project_stats(qq,kk,z,wh),rep=100),postprocess_float_ms=tr.testing.do_bench_cudagraph(lambda:run(wf,kv),rep=100),postprocess_half_ms=tr.testing.do_bench_cudagraph(lambda:run(wh,kv),rep=100),kvar_ms=tr.testing.do_bench_cudagraph(lambda:kvar(pk),rep=100),fused_kvar_ms=tr.testing.do_bench_cudagraph(lambda:fused_kvar(pk),rep=100))
                rows.append(row);atomic(root/'checks.json',dict(status='running',rows=rows));print(json.dumps(row),flush=True)
            return old(l,q,pk,base,detail,cov,vvar)
        rt.decision=decision;emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):generate_chunk(engine,emb,101,c)
        assert len(seen)==3 and not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,scope='Transient probability compression and variance fusion tested separately. Full generation validation required.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
