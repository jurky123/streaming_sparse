import traceback
import torch
from ..common import config,atomic_json
from ..opt18.setup import ROOT as P
from ..opt17.data import Events
from ..opt17.model import JointGroupedRouter
from ..opt18.measure import Runner,Opt14
from ..opt16.grouping import order,gather
from ..opt9.segmented_attention import attention as old_attention
from .probe import ROOT,measure,nmse
from .cache import Cache
from .attention import Quantized,attention
from . import tile

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    _,m=config(P);ev=Events(m,24)
    x=next(x for x in ev.val if x['row']['history_tokens']==14040 and x['row']['step']==3)
    model=JointGroupedRouter(m,24,'Fast35').cuda()
    model.load_state_dict(torch.load(P/'training/layer24/Fast35/best.pt',weights_only=True)['model'])
    run=Runner(model,ev,x,Opt14(m,24));cache=Cache(run.pipeline,run.raw,'fp16')
    kv=cache.current(run.raw['kc'],run.raw['vc']);q=gather(run.q,order(run.q,model.tree('q')))
    quant=Quantized(cache);quant.current();rows=[]
    for g in [16,32]:
        ids=tile.route(q,kv.state,.35,8,g,.25)
        mask=tile.expand(ids,4680,296,g);n=kv.history_tokens
        ref=lambda:old_attention(q,kv.k[:,:n],kv.v[:,:n],kv.k[:,n:],kv.v[:,n:],mask,group=g,pack_v=True,fuse_pack=True)
        y=ref();baseline_ms=measure(ref)
        qi=torch.arange(128,device=q.device)*31;ki=torch.arange(kv.k.shape[1],device=q.device);pages=ki//4680*74+ki%4680//64
        exact=torch.nn.functional.scaled_dot_product_attention(q[:,qi][None],kv.k[None],kv.v[None],attn_mask=mask[:,qi][:,:,pages][None])[0]
        for mode in ['int8','bf16']:
            for splits,warps,stages in [(1,4,2),(1,4,3),(1,8,2),(2,4,2),(2,2,2),(1,2,2)]:
                try:
                    quantizer=quant if mode=='int8' else None
                    fn=lambda:attention(q,kv,ids,g,quantizer,splits,warps,stages)
                    out=fn();assert torch.isfinite(out).all()
                    err=nmse(out,y)
                    if mode=='int8':assert err<1e-6,err
                    else:assert nmse(out[:,qi],exact)<1e-4
                    row=dict(group=g,mode=mode,splits=splits,warps=warps,stages=stages,ms=measure(fn),old_ms=baseline_ms,nmse_vs_old=err,nmse_vs_same_mask=nmse(out[:,qi],exact))
                except Exception as e:traceback.print_exc();row=dict(group=g,mode=mode,splits=splits,warps=warps,stages=stages,error=str(e))
                rows.append(row);atomic_json(ROOT/'attention_probe.json',rows);print(row,flush=True)
if __name__=='__main__':main()
