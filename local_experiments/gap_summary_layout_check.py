import torch,types,json
from .broad_common import Engine,ROOT as DATA
from .gap_input_runtime import GapInputRuntime
from .gap_summary_layout_ops import layout_pool
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
            for name in ['before','after']:
                m.pool=old if name=='before' else types.MethodType(layout_pool,m)
                rt.pool_graphs={};out=rt.current_summary(l)
                exact=[torch.equal(x,y) for x,y in zip(out,ref)]
                errors=[float((x.float()-y.float()).abs().max()) for x,y in zip(out,ref)]
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                for _ in range(300):rt.current_summary(l)
                end.record();end.synchronize()
                row=dict(layer=l,variant=name,exact=exact,errors=errors,ms=start.elapsed_time(end)/300);rows.append(row);print(json.dumps(row),flush=True)
            m.pool=old
        atomic('artifacts/gap_broad_opt_20261031/summary_layout_checks.json',dict(rows=rows))
    finally:rt.close()
if __name__=='__main__':main()
