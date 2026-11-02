import torch,triton as tr,json
from .gap_route_bf16_stats import project_stats as reference
from .gap_route_fused_tuned_runtime import project_stats
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    assert not device_peers();torch.manual_seed(731);torch.set_num_threads(4);rows=[]
    q=torch.randn(12,1152,128,device='cuda',dtype=torch.bfloat16).float()
    for p in [216,288,360,410]:
        k=torch.randn(12,p*4,128,device='cuda')*3
        z=torch.empty(12,1152,p,device='cuda');w=torch.empty(12,1152,p*4,device='cuda');zr=torch.empty_like(z);wr=torch.empty_like(w)
        reference(q,k,zr,wr);project_stats(q,k,z,w)
        row=dict(p=p,z_exact=torch.equal(z,zr),w_exact=torch.equal(w,wr),reference_ms=tr.testing.do_bench_cudagraph(lambda:reference(q,k,zr,wr),rep=100),tuned_ms=tr.testing.do_bench_cudagraph(lambda:project_stats(q,k,z,w),rep=100));rows.append(row);assert row['z_exact'] and row['w_exact'];print(json.dumps(row),flush=True)
    assert not device_peers();atomic('artifacts/gap_route_stats_20261101/tiles/shape_checks.json',dict(status='complete',rows=rows,scope='Synthetic shape coverage; actual three-layer fidelity separately tested.'))
if __name__=='__main__':main()
