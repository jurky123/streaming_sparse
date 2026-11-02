import torch,types,json
from .broad_common import Engine,ROOT as DATA
from .gap_summary_lite_runtime import GapSummaryLiteRuntime as GapInputRuntime
from .gap_feature_pool_ops import feature_pool as raw_pool
from .gap_summary_layout_ops import layout_pool
from .gap_direct_summary_ops import direct_pool
from .gap_direct_summary_runtime import GapDirectSummaryRuntime
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    assert not device_peers();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();rt=GapInputRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
    rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
    rows=[]
    try:
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(6):generate_chunk(engine,emb,101,c)
        for l in [1,14,29]:
            m=rt.model[l];old=m.pool
            ref=[x.clone() for x in rt.current_summary(l)]
            for name in ['before','raw']:
                m.pool=old if name=='before' else types.MethodType(raw_pool,m)
                rt.pool_graphs={};call=GapInputRuntime.current_summary;out=call(rt,l)
                exact=[torch.equal(x,y) for x,y in zip(out,ref)]
                errors=[float((x.float()-y.float()).abs().max()) for x,y in zip(out,ref)]
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(300):call(rt,l)
                end.record();end.synchronize()
                row=dict(layer=l,variant=name,exact=exact,errors=errors,ms=start.elapsed_time(end)/300);rows.append(row);print(json.dumps(row),flush=True)
            m.pool=old
        atomic('artifacts/gap_feature_pool_20261101/checks.json',dict(rows=rows))
    finally:rt.close()
if __name__=='__main__':main()
