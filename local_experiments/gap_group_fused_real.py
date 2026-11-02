"""Real layer features, whole chunk and within-page hard grouping audit."""
import torch,triton as tr,json
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_summary_lite_runtime import GapSummaryLiteRuntime
from .gap_gather_runtime import balanced_gather as native
from .gap_group_fused import balanced_fused
from .cache_grouping import reorder
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,RunLock,disk_guard
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    root=Path('artifacts/gap_group_fused_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapSummaryLiteRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(6):generate_chunk(engine,emb,101,c)
        rows=[]
        for l in [1,14,29]:
            m=rt.model[l];p=rt.n[l];cache=engine.cache[l]
            k=cache['k'][0,p*64:p*64+4608].transpose(0,1).float();v=cache['v'][0,p*64:p*64+4608].transpose(0,1).float()
            x=m.group_features(k[:,None],v[:,None]).squeeze(1)
            def full(fn):
                order=fn(x,[3,3,2,2,2]);xp=reorder(x,order)
                inner=fn(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
                return order.reshape(12,72,64).gather(2,inner)[:,rt.page_order]
            ref=full(native);out=full(lambda x,f:balanced_fused(x,f,threshold=512))
            row=dict(layer=l,exact=torch.equal(ref,out),slot_agreement=float((ref==out).float().mean()),page_set_agreement=float((ref.sort(-1).values==out.sort(-1).values).all(-1).float().mean()),native_ms=tr.testing.do_bench_cudagraph(lambda:full(native),rep=150),fused_ms=tr.testing.do_bench_cudagraph(lambda:full(lambda x,f:balanced_fused(x,f,threshold=512)),rep=150))
            rows.append(row);print(json.dumps(row),flush=True);atomic(root/'real_checks.json',dict(status='running',rows=rows))
        assert not device_peers();atomic(root/'real_checks.json',dict(status='complete',rows=rows,scope='Three real-layer feature sets; not full-video quality acceptance.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
