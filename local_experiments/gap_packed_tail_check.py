import torch,triton as tr,json
from .gap_flat_tail import _decode,_flat_tail
from .gap_packed_tail import _packed_tail,pack
from .high_sparse_runtime import _budget_mask
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(739)
    h,n,d,p=12,4608,128,410;rand=lambda *s:torch.randn(*s,device='cuda')
    q=rand(1,n,h,d).bfloat16();pk=rand(h,p,4,d)*.2;base=rand(h,p,8).bfloat16();detail=(rand(h,p,3,8)*.1).bfloat16();pv=torch.empty(h,p,4,8,device='cuda')
    kh=torch.empty_like(pk);kl=torch.empty_like(pk);vh=torch.empty_like(pv);vl=torch.empty_like(pv)
    lut=rand(h,n//64,p).topk(41,-1).indices.sort(-1).values[None].int();bits=torch.empty(h,n//64,16,device='cuda',dtype=torch.uint32)
    _budget_mask[(n//64,h)](lut,bits,n//64,41,tr.next_power_of_2(41));scale=torch.zeros(h,device='cuda')
    out=torch.empty(h,n,8,device='cuda');lse=torch.empty(h,n,device='cuda');rows=[]
    def native():
        _decode[(tr.cdiv(pv.numel(),256),)](base,detail,pv,pv.numel(),256,enable_fp_fusion=False)
        return _flat_tail[(n//64,h)](q,pk,pv,bits,out,lse,scale,p,n,h,d,64,64,num_warps=4,num_stages=1)
    native();ref=(out.clone(),lse.clone());native_ms=tr.testing.do_bench_cudagraph(native,rep=200)
    for a,b in [(True,False),(False,True),(True,True)]:
        def run():
            _decode[(tr.cdiv(pv.numel(),256),)](base,detail,pv,pv.numel(),256,enable_fp_fusion=False)
            if a:pack(pk,kh,kl)
            if b:pack(pv,vh,vl)
            return _packed_tail[(n//64,h)](q,pk,pv,bits,out,lse,scale,kh,kl,vh,vl,a,b,p,n,h,d,64,64,num_warps=4,num_stages=1)
        k=run();row=dict(pack_k=a,pack_v=b,exact=torch.equal(out,ref[0]) and torch.equal(lse,ref[1]),error=float((out-ref[0]).abs().max()),native_ms=native_ms,ms=tr.testing.do_bench_cudagraph(run,rep=200),registers=k.n_regs,spills=k.n_spills);rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_broad_opt_20261031/packed_tail_checks.json',dict(rows=rows))
if __name__=='__main__':main()
