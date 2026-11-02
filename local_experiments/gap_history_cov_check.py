"""Real cache data: covariance fidelity and graph timing on three layers."""
import json,torch,triton as tr
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_combined_runtime import GapRouteCombinedRuntime as GapRouteTensorRuntime
from .gap_covariance_kernel import covariance
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,RunLock,disk_guard
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    root=Path('artifacts/gap_history_cov_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapRouteTensorRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(6):generate_chunk(engine,emb,101,c)
        rows=[]
        for l in [1,14,29]:
            m=rt.model[l];cache=engine.cache[l];p=rt.n[l]
            for region,first in [('history',0)]:
                k=cache['k'][0,first*64:first*64+4608].transpose(0,1).reshape(12,72,64,128)
                v=cache['v'][0,first*64:first*64+4608].transpose(0,1).reshape_as(k)
                k=k.float();assert torch.equal(k,k.bfloat16().float())
                ref=m.covariance(k,v);out=covariance(k,v,m.basis)
                row=dict(layer=l,region=region,k_dtype=str(k.dtype),bf16_roundtrip_exact=True,exact=torch.equal(out,ref),mismatch_fraction=float((out!=ref).float().mean()),nmse=float((out.float()-ref.float()).square().sum()/ref.float().square().sum()),max_error=float((out.float()-ref.float()).abs().max()),native_ms=tr.testing.do_bench_cudagraph(lambda:m.covariance(k,v),rep=200),candidate_ms=tr.testing.do_bench_cudagraph(lambda:covariance(k,v,m.basis),rep=200))
                rows.append(row);print(json.dumps(row),flush=True);atomic(root/'checks.json',dict(status='running',rows=rows))
        assert not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,scope='Real-layer current and history covariance; no full-video claim.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
