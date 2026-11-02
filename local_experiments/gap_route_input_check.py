import torch,triton,json
from .gap_route_input_kernels import query,kvar
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.manual_seed(743);rows=[]
    q=torch.randn(1,4608,12,128,device='cuda').bfloat16();offsets=torch.linspace(0,63,16,device='cuda').round().long()
    def native_q():return q[0].transpose(0,1).float().reshape(12,72,64,128).index_select(2,offsets)
    a=native_q();b=query(q,offsets);rows.append(dict(name='query',exact=torch.equal(a,b),original_stride=list(a.stride()),new_stride=list(b.stride()),native_ms=triton.testing.do_bench_cudagraph(native_q,rep=200),ms=triton.testing.do_bench_cudagraph(lambda:query(q,offsets),rep=200)))
    for p in [216,410]:
        k=torch.randn(12,p,4,128,device='cuda')
        def native_k():return (k-k.mean(2,keepdim=True)).square().mean((2,3)).log1p()
        a=native_k();b=kvar(k);rows.append(dict(name='kvar',p=p,exact=torch.equal(a,b),max_error=float((a-b).abs().max()),native_ms=triton.testing.do_bench_cudagraph(native_k,rep=200),ms=triton.testing.do_bench_cudagraph(lambda:kvar(k),rep=200)))
    atomic('artifacts/gap_route_input_20261031/input_checks.json',dict(rows=rows));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
