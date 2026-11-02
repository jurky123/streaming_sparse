import json,torch,triton as tr
from .gap_pool_reduce import weighted as reference
from .gap_pool_linear import weighted
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    assert not device_peers();torch.manual_seed(739);torch.set_num_threads(4);rows=[]
    for dtype in [torch.bfloat16,torch.float32]:
        for layout in ['token_head','contiguous']:
            k=torch.randn(4608,12,128,device='cuda',dtype=dtype).transpose(0,1).reshape(12,72,64,128);v=torch.randn_like(k)
            if layout=='contiguous':k=k.contiguous();v=v.contiguous()
            a=torch.randn(12,72,4,16,device='cuda').softmax(-1);ref=reference(k,v,a,block=32)
            for b,w in [(32,4),(64,4),(128,4),(128,8)]:
                out=weighted(k,v,a,block=b,warps=w);row=dict(dtype=str(dtype),layout=layout,block=b,warps=w,exact=all(torch.equal(x,y) for x,y in zip(ref,out)),reference_ms=tr.testing.do_bench_cudagraph(lambda:reference(k,v,a,block=32),rep=100),ms=tr.testing.do_bench_cudagraph(lambda:weighted(k,v,a,block=b,warps=w),rep=100));rows.append(row);assert row['exact'];print(json.dumps(row),flush=True)
    assert not device_peers();atomic('artifacts/gap_pool_linear_20261101/checks.json',dict(status='complete',rows=rows,scope='Synthetic native shapes and storage types; real-layer pool and video checks still required.'))
if __name__=='__main__':main()
