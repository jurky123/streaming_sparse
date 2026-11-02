"""Local empirical throughput references, not hardware counter roofline proof."""
import torch,triton,json
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    a=torch.ones(2**26,device='cuda');b=torch.empty_like(a)
    ms=triton.testing.do_bench_cudagraph(lambda:b.copy_(a),rep=500)
    rows.append(dict(operation='contiguous device copy, 256MiB payload, read+write counted',ms=ms,bytes=2*a.numel()*a.element_size(),effective_TBps=2*a.numel()*a.element_size()/ms/1e9));del a,b
    for dtype,tf32 in [(torch.float32,False),(torch.bfloat16,False),(torch.float32,True)]:
        torch.backends.cuda.matmul.allow_tf32=tf32
        a=torch.randn(4096,4096,device='cuda',dtype=dtype);b=torch.randn_like(a);c=torch.empty_like(a)
        ms=triton.testing.do_bench_cudagraph(lambda:torch.mm(a,b,out=c),rep=500)
        rows.append(dict(operation='4096 square cuBLAS GEMM',dtype=str(dtype),tf32_allowed=tf32,ms=ms,gflops=2*4096**3/1e9,effective_TFLOPs=2*4096**3/ms/1e9));del a,b,c
    assert not device_peers()
    result=dict(device=torch.cuda.get_device_name(),rows=rows,caveat='Empirical large regular-workload references, not theoretical peaks or attainable bounds for small irregular kernels. TF32 setting recorded per GEMM. Copies count read+write, no counters or direct HBM/L2 traffic measurement.')
    atomic('artifacts/gap_route_stats_20261031/calibration.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
