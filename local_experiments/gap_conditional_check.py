import torch,triton,json
from .gap_conditional_kernel import conditional
from .conditional_v_kernels import HAAR
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(712);rows=[]
    haar=torch.tensor(HAAR,device='cuda',dtype=torch.float32)
    for p in [216,410]:
        w=torch.randn(12,72,p,4,device='cuda').softmax(-1);base=torch.randn(12,p,8,device='cuda').bfloat16();detail=torch.randn(12,p,3,8,device='cuda').bfloat16()
        def native():
            c=torch.einsum('htpr,ar->htpa',w,haar)
            return base.float()[:,None]+torch.einsum('htpa,hpad->htpd',c,detail.float()),c
        ref,c=native()
        for pair in [False,True]:
            out,cc=conditional(w,base,detail,pair=pair,debug=True)
            row=dict(p=p,pair=pair,coeff_exact=torch.equal(c,cc),output_exact=torch.equal(ref,out),coeff_error=float((c-cc).abs().max()),output_error=float((ref-out).abs().max()),native_stride=list(ref.stride()),new_stride=list(out.stride()),native_ms=triton.testing.do_bench_cudagraph(native,rep=200),ms=triton.testing.do_bench_cudagraph(lambda:conditional(w,base,detail,pair=pair),rep=200));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_input_20261031/conditional_checks.json',dict(rows=rows))
if __name__=='__main__':main()
