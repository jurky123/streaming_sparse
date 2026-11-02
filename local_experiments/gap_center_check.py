import json,torch,triton
from .gap_center_kernel import centers
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(814);rows=[]
    for layout in ['token_head','contiguous']:
        for dtype in ['bf16','float32']:
            k=torch.randn(1,4608,12,128,device='cuda')[0].transpose(0,1).reshape(12,72,64,128);v=torch.randn_like(k)
            if dtype=='bf16':k=k.bfloat16().float();v=v.bfloat16().float()
            if layout=='contiguous':k=k.contiguous();v=v.contiguous()
            kg=k.reshape(12,72,4,16,128);vg=v.reshape_as(kg)
            def reference():
                km=k.mean(2,keepdim=True);vm=v.mean(2,keepdim=True)
                return km,vm,k-km,kg-kg.mean(3,keepdim=True),vg-vg.mean(3,keepdim=True)
            target=reference()
            for b in [32,64]:
                out=centers(k,v,b);row=dict(layout=layout,dtype=dtype,block=b,exact=[torch.equal(x,y) for x,y in zip(target,out)],errors=[float((x-y).abs().max()) for x,y in zip(target,out)],strides=[list(x.stride()) for x in out])
                row['ms']=triton.testing.do_bench_cudagraph(lambda:centers(k,v,b),rep=200);row['native_ms']=triton.testing.do_bench_cudagraph(reference,rep=200)
                rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_pool_20261031/center_checks.json',dict(rows=rows,passed=all(all(r['exact']) for r in rows)))
if __name__=='__main__':main()
