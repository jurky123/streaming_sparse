"""Opt19 block interface. Explicit history lifetime; no whole-model default change."""
from ..opt16.grouping import order,gather,restore
from .cache import Cache
from .attention import Quantized,attention,native
from .route_fast import scores
from .tile import select
from .validate import native_mask

class Pipeline:
    def __init__(self,model,raw,group=32,budget=.25,mode='shared',precision='fp16'):
        assert mode in ('native','shared')
        assert model.pipeline is not None,'Initialize the model pipeline before creating an owned history snapshot'
        self.model=model;self.group=group;self.budget=budget;self.mode=mode
        self.cache=Cache(model.pipeline,raw,precision)
        self.quant=Quantized(self.cache)
        self.fraction=model.options['candidate_fraction'];self.reps=model.options['representatives']

    def refresh_history(self,k,v):
        self.cache.refresh_history(k,v);self.quant.history()

    def run(self,q,kc,vc,return_aux=False):
        kv=self.cache.current(kc,vc);self.quant.current()
        ix=order(q,self.model.tree('q'));qq=gather(q,ix)
        fine,candidates=scores(qq,kv.state,self.fraction,self.reps)
        b=kv.state['key'].shape[1]
        if self.mode=='native':
            chosen=native_mask(fine,candidates,b)
            out=native(qq,kv,chosen,self.quant,self.group)
        else:
            chosen=select(fine,candidates,b,self.group,self.budget)
            # The split2/warp2 path is numerically identical to the old full-mask
            # attention on the probe, and cheaper than constructing that mask.
            out=attention(qq,kv,chosen,self.group,self.quant,splits=2,warps=2,stages=2)
        out=restore(out,ix)
        return (out,dict(selection=chosen,kv=kv,q_order=ix,fine=fine,candidates=candidates)) if return_aux else out
