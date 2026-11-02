import torch,triton,json
from .gap_feature_kernel import _features_conditional
from .gap_feature_tiled import _features_tiled
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(89);rows=[]
    for p in [216,410]:
        h,t=12,72;mass=torch.rand(h,t,p,device='cuda');mass=mass/mass.sum(-1,keepdim=True)
        peak=mass*2;code=torch.randn(h,t,p,8,device='cuda');vv=torch.rand(h,p,device='cuda');kv=torch.rand_like(vv);w=torch.randn(h,14,device='cuda');mask=mass>.003;den=torch.rand(h,t,1,device='cuda')+.1;sel=torch.randn(h,t,8,device='cuda');av=torch.randn_like(sel);ratio=mass/den;spread=torch.rand_like(mass)
        args=(peak,mass,code,vv,kv,w,mask,den,sel,av,ratio,spread);ref=torch.empty(h,t,p,14,device='cuda');out=torch.empty_like(ref)
        def native():_features_conditional[(h*t,)](*args,ref,p,triton.next_power_of_2(p),*code.stride()[:3],num_warps=4,enable_fp_fusion=False)
        native()
        for b in [32,64,128,256]:
            def fused():_features_tiled[(h*t,triton.cdiv(p,b))](*args,out,p,b,*code.stride()[:3],num_warps=4,enable_fp_fusion=False)
            fused();row=dict(p=p,block=b,exact=torch.equal(ref,out),error=float((ref-out).abs().max()),native_ms=triton.testing.do_bench_cudagraph(native,rep=100),ms=triton.testing.do_bench_cudagraph(fused,rep=100));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_memory_20261031/feature_tiled_checks.json',dict(rows=rows))
if __name__=='__main__':main()
