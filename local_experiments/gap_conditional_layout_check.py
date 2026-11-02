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
        for native_layout in [True,False]:
            pair=False
            out,cc=conditional(w,base,detail,pair=pair,debug=True,native_layout=native_layout)
            mass=torch.rand(12,72,p,device='cuda');mask=mass>.5
            def agg(v):return (mass[...,None]*v).sum(2),(mass[...,None]*mask[...,None]*v).sum(2)
            agga=agg(ref);aggb=agg(out)
            row=dict(aggregation_exact=[torch.equal(a,b) for a,b in zip(agga,aggb)],p=p,native_layout=native_layout,pair=pair,coeff_exact=torch.equal(c,cc),output_exact=torch.equal(ref,out),coeff_error=float((c-cc).abs().max()),output_error=float((ref-out).abs().max()),native_stride=list(ref.stride()),new_stride=list(out.stride()),native_ms=triton.testing.do_bench_cudagraph(native,rep=200),ms=triton.testing.do_bench_cudagraph(lambda:conditional(w,base,detail,pair=pair,native_layout=native_layout),rep=200));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_input_20261031/conditional_layout_checks.json',dict(rows=rows))
if __name__=='__main__':main()
