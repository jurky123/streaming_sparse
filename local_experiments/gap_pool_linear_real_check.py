import torch,json,triton as tr
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_fused_tuned_runtime import GapRouteFusedTunedRuntime
from .gap_pool_linear_runtime import vector_weighted,current_ops,history_ops
from .rollout_suite import generate_chunk
from .gap_group_isolated import device_peers
from .learned_cache.io import atomic,RunLock,disk_guard
@torch.no_grad()
def main():
    root=Path('artifacts/gap_pool_linear_20261101');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapRouteFusedTunedRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16);rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(6):generate_chunk(engine,emb,101,c)
        rows=[]
        for l in [1,14,29]:
            p=rt.n[l];cache=engine.cache[l];k=cache['k'][0,p*64:p*64+4608].transpose(0,1).reshape(12,72,64,128);v=cache['v'][0,p*64:p*64+4608].transpose(0,1).reshape_as(k);m=rt.model[l]
            for cheap in [True,False]:
                kk,vv=(k,v) if cheap else (k.float().contiguous(),v.contiguous())
                module=current_ops if cheap else history_ops;old=module.weighted
                def run():return m.pool(kk,vv,cheap=cheap)
                ref=tuple(x.clone() for x in run());before=tr.testing.do_bench_cudagraph(run,rep=100)
                try:
                    module.weighted=vector_weighted;out=run();exact=[torch.equal(x,y) for x,y in zip(ref,out)];after=tr.testing.do_bench_cudagraph(run,rep=100)
                    assert all(exact)
                finally:module.weighted=old
                row=dict(layer=l,cheap=cheap,exact=exact,before_ms=before,after_ms=after);rows.append(row);atomic(root/'real_checks.json',dict(status='running',rows=rows));print(json.dumps(row),flush=True)
        assert not device_peers();atomic(root/'real_checks.json',dict(status='complete',rows=rows,scope='Actual model layer pool inputs; history uses the same tokens in contiguous storage before reordering. Full rollout still required.'))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
