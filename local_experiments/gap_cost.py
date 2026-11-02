"""Calibrated compute/bandwidth floors and separate real-shape stage timings."""
import json,math,time,types,torch,triton
from .gap_model import GapLayer
from .gap_cached_ops import cached_route as route,cached_pool
from .conditional_v_kernels import HAAR
from .conditional_v_kernels import _conditional_tail
from .compensation_kernels import _merge
from .high_sparse_runtime import _budget_mask
from .cache_sparse_prototype import load_kernel
from .all_sparse_attention import native
from .learned_cache.io import atomic
ROOT='artifacts/four_gaps_20261031/'
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(819);torch.backends.cuda.matmul.allow_tf32=False
    def rand(*s):return torch.randn(*s,device='cuda')
    def bench(fn):return triton.testing.do_bench_cudagraph(fn,rep=300)
    calibration={}
    for dtype,tf32,name in [(torch.bfloat16,False,'bf16_tensor'),(torch.float32,False,'fp32_strict'),(torch.float32,True,'tf32_tensor')]:
        torch.backends.cuda.matmul.allow_tf32=tf32;a=rand(4096,4096).to(dtype);b=rand(4096,4096).to(dtype);c=torch.empty_like(a)
        ms=bench(lambda:torch.mm(a,b,out=c));calibration[name]=dict(ms=ms,tflops=2*4096**3/(ms*1e9));del a,b,c
    torch.backends.cuda.matmul.allow_tf32=False
    a=torch.zeros(64*1024*1024,device='cuda',dtype=torch.bfloat16);b=torch.empty_like(a);ms=bench(lambda:b.copy_(a));bandwidth=2*a.numel()*a.element_size()/(ms*1e6);del a,b
    n,h,d,p,keep,m=4608,12,128,410,41,16
    q=rand(1,n,h,d).bfloat16();k=rand(1,p*64,h,d).bfloat16();v=rand(1,p*64,h,d).bfloat16();pk=(rand(h,p,4,d)*.2).contiguous();base=rand(h,p,8).bfloat16();detail=(rand(h,p,3,8)*.1).bfloat16();cov=(rand(h,p,8)*.1).bfloat16();vv=rand(h,p).abs()
    rt=types.SimpleNamespace(bank={1:rand(h,32,d)},weights={1:rand(h,4,d)*.01},basis={1:torch.linalg.qr(rand(h,d,8))[0].bfloat16()},centers={1:rand(h,d)*.1},sw={1:rand(h,14)*.01})
    layer=GapLayer(rt,1);layer._cached_haar=torch.tensor(HAAR,device="cuda",dtype=torch.float32);layer.pool=types.MethodType(cached_pool,layer);offsets=torch.linspace(0,63,m,device='cuda').round().long();lut=route(layer,q,pk,base,detail,cov,vv,offsets,False)[0]
    out=torch.empty_like(q);lse=torch.empty(1,h,n,device='cuda');kernel=load_kernel()
    def exact():kernel[(n//64,h)](q,k,v,d**-.5,keep,lut,lse,out,h,n,k.shape[1],n//64,d,64,64,num_warps=4,num_stages=3)
    exact();bits=torch.empty(h,n//64,16,device='cuda',dtype=torch.uint32);_budget_mask[(n//64,h)](lut,bits,n//64,keep,triton.next_power_of_2(keep))
    tail=torch.empty(h,n,8,device='cuda');lt=torch.empty(h,n,device='cuda');merged=torch.empty_like(q);ones=torch.ones(h,device='cuda')
    def tail_fn():_conditional_tail[(n//64,h)](q,pk,base,detail,lut,bits,tail,lt,layer.comp_logscale,p,n,h,d,3,64,16,num_warps=4,num_stages=1)
    tail_fn()
    def merge():_merge[(triton.cdiv(n,16),h)](out,lse,tail,lt,layer.basis.bfloat16(),layer.center,ones,merged,n,h,d,16,num_warps=4)
    curk=k[0,-4608:].transpose(0,1).reshape(h,72,64,d);curv=v[0,-4608:].transpose(0,1).reshape_as(curk)
    query=q[0].transpose(0,1).float().reshape(h,72,64,d).index_select(2,offsets).reshape(h,72*m,d);raw=torch.empty(h,72*m,p*4,device='cuda')
    functions={'dense_sdpa':lambda:torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)), 'exact_selected':exact,'conditional_tail':tail_fn,'tail_merge':merge,'native_exact_plus_tail':lambda:native(q,k,v,pk,base,detail,layer.basis.bfloat16(),layer.center,layer.comp_logscale,lut,kernel),'route_complete_graph':lambda:route(layer,q,pk,base,detail,cov,vv,offsets,False),'route_qk_only':lambda:torch.bmm(query,pk.flatten(1,2).transpose(1,2),out=raw),'current_pool_cheap':lambda:layer.pool(curk,curv,cheap=True),'current_pool_full':lambda:layer.pool(curk,curv),'current_covariance':lambda:layer.covariance(curk,curv)}
    timings={}
    for name,fn in functions.items():timings[name]=bench(fn);print(name,timings[name],flush=True)
    dense=4*n*h*p*64*d;exact_flops=4*n*h*keep*64*d;tail_qk=2*n*h*p*4*d;tail_v=2*n*h*p*4*8;route_qk=2*h*(n//64)*m*p*4*d
    # These are calibrated optimistic floors, not achieved hardware roofline proof.
    floors={'dense_ms':dense/(calibration['bf16_tensor']['tflops']*1e9),'exact_selected_ms':exact_flops/(calibration['bf16_tensor']['tflops']*1e9),'route_qk_ms':route_qk/(calibration['fp32_strict']['tflops']*1e9),'tail_ideal_math_ms':(tail_qk+tail_v)/(calibration['tf32_tensor']['tflops']*1e9),'tail_split_and_padding_approx_ms':(2*tail_qk+4*tail_v)/(calibration['tf32_tensor']['tflops']*1e9)}
    result=dict(device=torch.cuda.get_device_name(),calibration=calibration,copy_bandwidth_GBps=bandwidth,timings_ms=timings,flops=dict(dense=dense,exact=exact_flops,tail_qk=tail_qk,tail_v=tail_v,route_qk=route_qk),ideal_attention_plus_tail_ratio=(exact_flops+tail_qk+tail_v)/dense,calibrated_optimistic_floors_ms=floors,scope='Synthetic steady-state actual shapes; CUDA graph removes Python launch gaps. Only matrix FLOPs counted; normalization, topk, residual features, memory, alignment and kernel occupancy not free. Not an end-to-end speedup prediction. TF32 split/padding count approximate, no hardware-counter proof of limit attainment.')
    atomic(ROOT+'cost.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
