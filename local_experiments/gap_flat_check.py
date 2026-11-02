import json,torch,triton
from .conditional_v_kernels import compensate as reference
from .gap_flat_tail import compensate
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(727);rows=[];n,h,d=4608,12,128
    def rand(*s):return torch.randn(*s,device='cuda')
    for p in [216,410]:
        q=rand(1,n,h,d).bfloat16();pk=rand(h,p,4,d)*.2;base=rand(h,p,8).bfloat16();detail=(rand(h,p,3,8)*.1).bfloat16();basis=torch.linalg.qr(rand(h,d,8))[0].bfloat16().contiguous();center=rand(h,d)*.1;gamma=torch.ones(h,device='cuda');scale=torch.zeros(h,device='cuda')
        lut=rand(h,n//64,p).topk(int(p*.1),-1).indices.sort(-1).values[None].int();initial=rand(1,n,h,d).bfloat16();lse=torch.full((1,h,n),11.,device='cuda')
        y,t,z=reference(q,pk,base,detail,lut,lut,initial.clone(),lse,basis,center,gamma,scale)
        output=initial.clone()
        def old():reference(q,pk,base,detail,lut,lut,output,lse,basis,center,gamma,scale)
        old_ms=triton.testing.do_bench_cudagraph(old,rep=300)
        for s in [0]:
            y2,t2,z2=compensate(q,pk,base,detail,lut,lut,initial.clone(),lse,basis,center,gamma,scale)
            err=float((t-t2).square().sum()/t.square().sum());le=float((z-z2).abs().max());assert err<1e-9 and le<2e-5,(p,s,err,le)
            nmse=float((y.float()-y2.float()).square().sum()/y.float().square().sum());assert nmse<1e-6,nmse
            def run():compensate(q,pk,base,detail,lut,lut,output,lse,basis,center,gamma,scale)
            ms=triton.testing.do_bench_cudagraph(run,rep=300);row=dict(pages=p,splits=s,reference_ms=old_ms,split_ms=ms,speedup=old_ms/ms,tail_nmse=err,lse_max_error=le,output_nmse=nmse,output_exact_fraction=float((y==y2).float().mean()))
            rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_hotspots_20261031/flat_tail_checks.json',dict(passed=True,rows=rows,scope='Algebraic representative formulation, synthetic numerical and speed only; real quality gate not yet accepted'))
if __name__=='__main__':main()
