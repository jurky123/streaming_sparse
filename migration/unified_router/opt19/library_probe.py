import argparse,math,traceback
import torch
from ..common import config,atomic_json
from ..opt18.setup import ROOT as PARENT
from ..opt18.measure import Runner,Opt14
from ..opt17.model import JointGroupedRouter
from ..opt17.data import Events
from ..opt16.grouping import order,gather,restore
from ..opt9.segmented_attention import attention
from .cache import Cache
from . import tile
from .library import FlashInfer,Flex,padded_kv
from .probe import ROOT,measure,nmse

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['flashinfer','flex'],required=True)
    p.add_argument('--precision',default='reference');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.cache_size_limit=256
    _,m=config(PARENT);events=Events(m,24);prompt=sorted({x['row']['prompt_id'] for x in events.val})[0]
    item=next(x for x in events.val if x['row']['prompt_id']==prompt and x['row']['history_tokens']==14040 and x['row']['step']==3)
    model=JointGroupedRouter(m,24,'Fast35').cuda().eval()
    model.load_state_dict(torch.load(PARENT/'training/layer24/Fast35/best.pt',map_location='cuda',weights_only=True)['model'])
    runner=Runner(model,events,item,Opt14(m,24));q,r,truth,targets=runner.q,runner.raw,runner.truth,runner.targets
    cache=Cache(runner.pipeline,r,a.precision);kv=cache.current(r['kc'],r['vc'])
    ix=order(q,model.tree('q'));qq=gather(q,ix);rows=[]
    for g in [16,32]:
        for budget in [.15,.20,.25]:
            try:
                print('compile',a.backend,g,budget,flush=True)
                b=kv.state['key'].shape[1];s=math.ceil(b*budget)
                lib=FlashInfer(12,q.shape[1],b,g,s) if a.backend=='flashinfer' else Flex(g)
                ids=tile.route(qq,kv.state,.35,8,g,budget,True)
                # Validate library output against the exact same sparse selection.
                mask=tile.expand(ids,q.shape[1],b,g)
                qi=torch.arange(128,device=q.device)*31
                ki=torch.arange(kv.k.shape[1],device=q.device);pages=ki//4680*74+ki%4680//64
                exact=torch.nn.functional.scaled_dot_product_attention(qq[:,qi][None],kv.k[None],kv.v[None],attn_mask=mask[:,qi][:,:,pages][None])[0]
                actual=lib.run(qq,kv,ids)
                err=nmse(actual[:,qi],exact);assert err<1e-4,(a.backend,g,budget,err)
                def run():
                    kv=cache.current(r['kc'],r['vc']);ix=order(q,model.tree('q'));qq=gather(q,ix)
                    ids=tile.route(qq,kv.state,.35,8,g,budget,True)
                    return restore(lib.run(qq,kv,ids),ix)
                y=run();pk,pv=padded_kv(kv)
                row=dict(backend=a.backend,precision=a.precision,group=g,budget=budget,
                         nmse=nmse(y[:,targets],truth),kernel_nmse_vs_same_mask=err,
                         warm_ms=measure(run),attention_with_pack_ms=measure(lambda:lib.run(qq,kv,ids)),
                         attention_prepared_ms=measure(lambda:lib.prepared(qq,pk,pv,ids)))
            except Exception as e:traceback.print_exc();row=dict(backend=a.backend,group=g,budget=budget,error=str(e))
            rows.append(row);print(row,flush=True);atomic_json(ROOT/f'library_{a.backend}_{a.precision}.json',dict(sample=item['row'],rows=rows))
if __name__=='__main__':main()
