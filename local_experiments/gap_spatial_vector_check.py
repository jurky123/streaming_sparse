"""Real history scores: serial versus vector spatial softmax and feature output."""
import torch,json,triton as tr
from pathlib import Path
from triton.runtime.errors import OutOfResources
from .broad_common import Engine,ROOT as DATA
from .gap_pool_linear_runtime import GapPoolLinearRuntime
from .gap_group_projection import group_features,project
from . import gap_group_projection as module
from .gap_spatial_vector import spatial
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,RunLock,disk_guard
@torch.no_grad()
def main():
    root=Path('artifacts/gap_spatial_vector_20261101');lock=RunLock(root);rt=None;original=module.spatial
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapPoolLinearRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16);rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(6):generate_chunk(engine,emb,101,c)
        rows=[]
        for l in [1,14,29]:
            p=rt.n[l];cache=engine.cache[l];k=cache['k'][0,p*64:p*64+4608].transpose(0,1).float();v=cache['v'][0,p*64:p*64+4608].transpose(0,1).float();m=rt.model[l]
            for pages in [1,72]:
                kk=k.reshape(12,pages,4608//pages,128);vv=v.reshape_as(kk);raw=project(kk,m.bank);score=(raw-raw.mean(-2,keepdim=True))/128**.5
                ref=original(score,block=4 if pages==1 else 8);ref_f=group_features(m,kk,vv)
                options=[(1,4),(1,8),(2,4),(2,8),(4,8)] if pages==1 else [(4,4),(8,4),(16,4),(32,4)]
                for b,w in options:
                    row=dict(layer=l,pages=pages,block=b,warps=w)
                    try:
                        out=spatial(score,b,w);row.update(max_error=float((out-ref).abs().max()),reference_ms=tr.testing.do_bench_cudagraph(lambda:original(score,block=4 if pages==1 else 8),rep=100),ms=tr.testing.do_bench_cudagraph(lambda:spatial(score,b,w),rep=100))
                        module.spatial=lambda x,block:spatial(x,b,w)
                        value=group_features(m,kk,vv);row.update(feature_nmse=float((value-ref_f).square().sum()/ref_f.square().sum()),feature_ms=tr.testing.do_bench_cudagraph(lambda:group_features(m,kk,vv),rep=100))
                    except OutOfResources as exc:row['unavailable']=str(exc)
                    finally:module.spatial=original
                    rows.append(row);atomic(root/'checks.json',dict(status='running',rows=rows));print(json.dumps(row),flush=True)
        assert not device_peers();atomic(root/'checks.json',dict(status='complete',rows=rows,scope='Actual scores, relative to raw group-projection candidate. Changed reduction tree and exp approximation; video validation still required.'))
    finally:
        module.spatial=original
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
