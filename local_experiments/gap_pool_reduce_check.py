import json,torch,triton
from .gap_pool_reduce import weighted
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(813);rows=[]
    for p in [2,72]:
        for layout in ['token_head','contiguous']:
            for scale in [.01,1.,100.]:
                k=(torch.randn(1,p*64,12,128,device='cuda').bfloat16()*scale)[0].transpose(0,1).reshape(12,p,64,128).float();v=torch.randn_like(k)
                if layout=='contiguous':k=k.contiguous();v=v.contiguous()
                a=torch.randn(12,p,4,16,device='cuda').softmax(-1)
                kg=k.reshape(12,p,4,16,128);vg=v.reshape_as(kg)
                x=(a[...,None]*kg).sum(-2);y=(a[...,None]*vg).sum(-2)
                for b in [32,64,128]:
                    u,w=weighted(k,v,a,b);row=dict(p=p,layout=layout,scale=scale,block=b,exact=torch.equal(x,u) and torch.equal(y,w),error=max(float((x-u).abs().max()),float((y-w).abs().max())))
                    if p==72 and scale==1.:
                        row['ms']=triton.testing.do_bench_cudagraph(lambda:weighted(k,v,a,b),rep=200)
                        row['native_ms']=triton.testing.do_bench_cudagraph(lambda:((a[...,None]*kg).sum(-2),(a[...,None]*vg).sum(-2)),rep=200)
                    rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_pool_20261031/reduce_checks.json',dict(rows=rows,passed=all(x['exact'] for x in rows)))
if __name__=='__main__':main()
