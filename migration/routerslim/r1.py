"""Opt-in, differentiable execution rewrites of the frozen v0.8 equations.

No parameter, dtype, page, loss, query or gradient policy changes. Derived
summaries live in the state built for ONE parameter version / autograd graph.
The original equations remain available for numerical-boundary fallback.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import torch
import torch.nn.functional as F

_originals = {}
_options = {}
stats = dict(boundary_groups=0, score_queries=0)


def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def rewrite(function, replacements=(), additions=None, namespace=None):
    source=textwrap.dedent(inspect.getsource(function))
    for old,new in replacements:
        assert source.count(old)==1,(function.__name__,old)
        source=source.replace(old,new)
    scope={**function.__globals__,**(namespace or {})}
    exec(compile(source,__file__,'exec'),scope)
    return scope[function.__name__]


def install(variance=True,projection=True,skip_zero_aux=True):
    from streaming_sparse.v08 import router,backbone
    from streaming_sparse.v08.execution import accelerate
    assert not _originals,'Install once per process; use uninstall for isolated comparisons'
    _options.update(variance=variance,projection=projection,skip_zero_aux=skip_zero_aux)
    _originals.update(raw=router.KSketch.raw,estimate=router.KSketch.estimate,
        sketch_build=router.KSketch.build,build=router.InformationRouter.build,
        score=router.InformationRouter.score,head=backbone.FunctionalBackbone._head)

    # A true legacy scoring path, independent of any installed class rewrites.
    legacy_estimate=rewrite(_originals['estimate'],
        [('self.raw(q,state)','_legacy_raw(self,q,state)')],namespace={'_legacy_raw':_originals['raw']})
    legacy_score=rewrite(_originals['score'],
        [('self.sketch.estimate(q,state,kc,vc if self.kind==\'V\' else None,self.kind)',
          '_legacy_estimate(self.sketch,q,state,kc,vc if self.kind==\'V\' else None,self.kind)')],
        namespace={'_legacy_estimate':legacy_estimate})
    _originals['legacy_score']=legacy_score
    if variance:
        original_sketch_build=_originals['sketch_build']
        def sketch_build(self,*args,**kwargs):
            state=original_sketch_build(self,*args,**kwargs)
            valid=state['count']>1e-8
            state['_r1_mean_variance']=(state['variance']*valid[...,None]).sum(1)/valid.sum(-1).clamp_min(1)[:,None]
            return state
        router.KSketch.build=sketch_build
        router.KSketch.raw=rewrite(_originals['raw'],[
            ("kvar=(torch.einsum('qd,bpd->qbp',direction.square(),state['variance'])*valid[None]).sum(-1)/prototype_count",
             "kvar=direction.square()@state['_r1_mean_variance'].T")])
    if projection:
        original_router_build=_originals['build']
        def build(self,*args,**kwargs):
            state=original_router_build(self,*args,**kwargs)
            weight=self.content[state['pair']].weight
            state['_r1_content_weight']=weight
            state['_r1_projected_content']=F.linear(state['key' if self.kind=='K' else 'value'],weight)
            return state
        router.InformationRouter.build=build
        router.KSketch.estimate=rewrite(_originals['estimate'],[
            ("return dict(raw_logz=raw", """low_mu=torch.einsum('qbp,bpr->qbr',w,state['_r1_projected_content'])
    low_oc=_linear(oc,state['_r1_content_weight'])
    low_dense=(mh[...,None]*low_mu.double()).sum(1)+mc[:,None]*low_oc.double()
    low_residual=(mh[...,None]*(low_mu.double()-low_dense[:,None])).float()
    return dict(_r1_projected_residual=low_residual,raw_logz=raw""")],namespace={'_linear':F.linear})
        router.InformationRouter.score=rewrite(_originals['score'],[
            ("projection=self.norm[pair](self.content[pair](r))",
             "projection=self.norm[pair](e['_r1_projected_residual']+self.content[pair].bias)")])

    # Only ambiguous groups revert to the ORIGINAL equations AND original GEMM
    # shapes. The existing absolute/relative guard is kept unchanged.
    source=textwrap.dedent(inspect.getsource(accelerate.head))
    needle='result=checkpoint(score_function,*args,use_reentrant=False) if torch.is_grad_enabled() else score_function(*args)'
    assert source.count(needle)==2
    pos=source.rfind(needle)
    source=source[:pos]+source[pos:].replace(needle,
        'result=checkpoint(legacy_function,*args,use_reentrant=False) if torch.is_grad_enabled() else legacy_function(*args)',1)
    marker='    # Query rows are independent.'
    source=source.replace(marker,"""    def legacy_function(q0,ck,cv,st):
        route=_legacy_score(self.router,q0,ck,cv,st,keep,step,detach_router,mass_only)
        return route['scores'],route['delta'].detach().square().sum()
    _stats['score_queries']+=len(q)
"""+marker)
    source=source.replace('    if groups:\n','    _stats[\'boundary_groups\']+=len(groups)\n    if groups:\n',1)
    if skip_zero_aux:
        # Calibration is still evaluated on exactly the original selected
        # tiles. Only the zero-coefficient attention-output auxiliary is pruned.
        source=source.replace('        if not detach_router:\n',"""        if getattr(self,'_routerslim_skip_att_aux',False):
            return q0.new_zeros((),dtype=torch.float32),self.router.sketch.calibration_loss(q0.detach(),hk.detach(),ck.detach(),st,e)
        if not detach_router:
""",1)
    scope={**accelerate.__dict__,'_legacy_score':legacy_score,'_stats':stats}
    exec(compile(source,__file__,'exec'),scope)
    backbone.FunctionalBackbone._head=scope['head']


def configure_update(network,update):
    # Original schedule: .1*max(0,1-(step-1)/500). Do not prune while nonzero.
    network._routerslim_skip_att_aux=bool(_options.get('skip_zero_aux') and update>=501)


def uninstall():
    if not _originals:return
    from streaming_sparse.v08 import router,backbone
    router.KSketch.raw=_originals['raw'];router.KSketch.estimate=_originals['estimate']
    router.KSketch.build=_originals['sketch_build'];router.InformationRouter.build=_originals['build']
    router.InformationRouter.score=_originals['score'];backbone.FunctionalBackbone._head=_originals['head']
    _originals.clear();_options.clear()
