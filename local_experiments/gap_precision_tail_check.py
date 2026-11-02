"""Resource-aware tiling of the flattened tail; independent of quality selection."""
import json,torch,triton as tr
from triton.runtime.errors import OutOfResources
from .gap_flat_tail import _decode,_flat_tail
from .gap_precision_tail import _precision_tail
from .high_sparse_runtime import _budget_mask
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(729)
    h,n,d,p,keep=12,4608,128,410,41
    rand=lambda *s:torch.randn(*s,device='cuda')
    q=rand(1,n,h,d).bfloat16();pk=rand(h,p,4,d)*.2;base=rand(h,p,8).bfloat16();detail=(rand(h,p,3,8)*.1).bfloat16();pv=torch.empty(h,p,4,8,device='cuda')
    _decode[(tr.cdiv(pv.numel(),256),)](base,detail,pv,pv.numel(),256,enable_fp_fusion=False)
    lut=rand(h,n//64,p).topk(keep,-1).indices.sort(-1).values[None].int();bits=torch.empty(h,n//64,16,device='cuda',dtype=torch.uint32)
    _budget_mask[(n//64,h)](lut,bits,n//64,keep,tr.next_power_of_2(keep));scale=torch.zeros(h,device='cuda');rows=[];ref=None
    for mode,bq,br,warps,stages in [(0,64,64,4,1),(1,64,64,4,1),(2,64,64,4,1),(3,64,64,4,1)]:
        if bq>64:
            rows.append(dict(bq=bq,br=br,warps=warps,stages=stages,invalid_reason="Crosses independent 64-query routing masks; unsupported"));continue
        out=torch.empty(h,n,8,device='cuda');lse=torch.empty(h,n,device='cuda')
        def run():
            if mode==0:return _flat_tail[(tr.cdiv(n,bq),h)](q,pk,pv,bits,out,lse,scale,p,n,h,d,bq,br,num_warps=warps,num_stages=stages)
            return _precision_tail[(tr.cdiv(n,bq),h)](q,pk,pv,bits,out,lse,scale,mode,p,n,h,d,bq,br,num_warps=warps,num_stages=stages)
        row=dict(mode=mode,bq=bq,br=br,warps=warps,stages=stages)
        try:
            compiled=run();torch.cuda.synchronize()
            if ref is None:ref=(out.clone(),lse.clone())
            row.update(exact=torch.equal(out,ref[0]) and torch.equal(lse,ref[1]),nmse=float((out-ref[0]).square().sum()/ref[0].square().sum()),lse_max=float((lse-ref[1]).abs().max()),registers=compiled.n_regs,shared_bytes=compiled.metadata.shared,spills=compiled.n_spills,ms=tr.testing.do_bench_cudagraph(run,rep=300))
        except OutOfResources as e:row['unavailable']=str(e)
        rows.append(row);print(json.dumps(row),flush=True)
        atomic('artifacts/gap_broad_opt_20261031/precision_tail_checks.json',dict(rows=rows,status='running'))
    assert not device_peers()
    atomic('artifacts/gap_broad_opt_20261031/precision_tail_checks.json',dict(rows=rows,status='complete',scope='Synthetic resource and speed sweep; not full-video acceptance.'))
if __name__=='__main__':main()
