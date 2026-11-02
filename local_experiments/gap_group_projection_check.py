"""Real history operands: feature projection cost and hard-group sensitivity."""
import torch,json,triton as tr
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_pool_linear_runtime import GapPoolLinearRuntime
from .gap_group_projection import group_features
from .gap_tensor_gram import balanced_tensor_gram
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,RunLock,disk_guard
@torch.no_grad()
def main():
    root=Path('artifacts/gap_group_projection_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapPoolLinearRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16);rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(6):generate_chunk(engine,emb,101,c)
        rows=[]
        for l in [1,14,29]:
            p=rt.n[l];cache=engine.cache[l];k=cache['k'][0,p*64:p*64+4608].transpose(0,1).float();v=cache['v'][0,p*64:p*64+4608].transpose(0,1).float();m=rt.model[l]
            assert torch.equal(k,k.bfloat16().float()) and torch.equal(v,v.bfloat16().float())
            for pages in [1,72]:
                kk=k.reshape(12,pages,4608//pages,128);vv=v.reshape_as(kk)
                ref=m.group_features(kk,vv);out=group_features(m,kk,vv)
                row=dict(layer=l,pages=pages,nmse=float((out-ref).square().sum()/ref.square().sum()),max_error=float((out-ref).abs().max()),reference_ms=tr.testing.do_bench_cudagraph(lambda:m.group_features(kk,vv),rep=100),candidate_ms=tr.testing.do_bench_cudagraph(lambda:group_features(m,kk,vv),rep=100))
                if pages==1:
                    a=balanced_tensor_gram(ref.squeeze(1),[3,3,2,2,2]);b=balanced_tensor_gram(out.squeeze(1),[3,3,2,2,2]);row['order_changed_fraction']=float((a!=b).float().mean())
                rows.append(row);atomic(root/'checks.json',dict(status='running',rows=rows));print(json.dumps(row),flush=True)
        assert not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,scope='Changed projection precision and centering order; hard grouping sensitivity recorded. No video or training acceptance yet.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
