"""Measured primitive floors and analytical QK/traffic lower bounds, not a proof of optimality."""
import json,statistics,subprocess,math
from pathlib import Path
import torch
from .route_graph_runtime import RouteGraph
from .route_norm_runtime import NormRouter
from .route_norm_kernels import logz
from .learned_cache.io import atomic
from .improve_accept import Telemetry
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(31)
    tele=Telemetry();tele.start();r=NormRouter();old=RouteGraph();p=410
    q=torch.randn(1,4608,12,128,device='cuda',dtype=torch.bfloat16);pk=torch.randn(12,p,4,128,device='cuda');c=torch.randn(12,p,8,device='cuda',dtype=torch.bfloat16);v=torch.ones(12,p,device='cuda');w=torch.randn(12,14,device='cuda')*.3
    r(q,pk,c,v,w,True);old(q,pk,c,v,w,True)
    raw=r.raw[:12*288*p*4].view(12,288,p*4);prob=logz(raw,p).softmax(-1);mass=prob.mean(2)
    big=torch.empty(128*1024*1024,device='cuda');dst=torch.empty_like(big)
    def measure(fn):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):out=fn()
        torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):out=fn()
        times=[]
        for _ in range(5):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(100):g.replay()
            b.record();b.synchronize();times.append(a.elapsed_time(b)/100)
        return dict(median_ms=statistics.median(times),reps_ms=times)
    stages={}
    for label,fn in [('qk_bmm',lambda:torch.bmm(r.query,pk.flatten(1,2).transpose(1,2))),('k_variance',lambda:(pk-pk.mean(2,keepdim=True)).square().mean((2,3)).log1p()),('native_logz',lambda:(raw.reshape(12,72,4,p,4)/math.sqrt(128)).logsumexp(-1)+math.log(16)),('fused_logz',lambda:logz(raw,p)),('softmax_and_mean',lambda:logz(raw,p).softmax(-1).mean(2)),('topk',lambda:mass.topk(41,-1)),('memory_copy_512MiB',lambda:dst.copy_(big))]:
        stages[label]=measure(fn)
    for label,router in [('old_graph',old),('norm_graph',r)]:
        times=[]
        for _ in range(5):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(100):router.graphs[(410,True)].replay()
            b.record();b.synchronize();times.append(a.elapsed_time(b)/100)
        stages[label]=dict(median_ms=statistics.median(times),reps_ms=times)
    props=torch.cuda.get_device_properties(0);load=tele.finish()
    # Copy counts both read and write; empirical streaming bandwidth, different cache regime from routing.
    bw=2*big.numel()*big.element_size()/(stages['memory_copy_512MiB']['median_ms']/1000)
    flops=2*12*288*(410*4)*128
    result=dict(gpu=props.name,sms=props.multi_processor_count,stages=stages,telemetry=load,qk_flops=flops,empirical_copy_bytes_per_second=bw,current_KV_min_read_bytes=2*4608*12*128*2,notes=['CUDA graphs, warmed medians; QK FP32/TF32 disabled.','old_graph is postprocess only; norm_graph includes QK/variance, so not identical stage boundaries.','Mandatory operation lower bounds omit launch/softmax/topk costs; not evidence of attained theoretical limit.'])
    atomic(Path('artifacts/route_kernel_20261031/roofline.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
