"""Batch independent queries; exact Page64 selection entirely on the GPU.

No lower precision, query subsampling, history truncation or objective changes.
Auxiliary queries keep their original 16-query grouping and seeded sample IDs.
Native hard attention keeps the original full-query, twelve-head layout.
"""
import hashlib
import math
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint

SCORE_BATCH = 128
GATE_BATCH = 128
GUARD_ABSOLUTE = 1e-4
GUARD_RELATIVE = 1e-6
_originals = {}


def identity():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode());h.update(path.read_bytes())
    return h.hexdigest()


def page_select(scores, costs, budget, block_ids=None):
    """Same stable descending scan for uniform pages plus one partial tail.

    A partial page consumes at most one full-page slot. If it fits in the final
    remainder it is selected wherever ranked; otherwise it must occur before
    the full-page quota is exhausted. Other cost layouts use the original scan.
    """
    original = _originals['select']
    if scores.ndim != 2 or costs.ndim != 1 or scores.shape[1] != len(costs):
        return original(scores, costs, budget, block_ids)
    if not len(costs):
        return torch.zeros_like(scores)
    # Costs are fixed metadata, not differentiable execution or router features.
    values = costs.detach().long()
    full = int(values[0])
    tail = int(values[-1])
    if full <= 0 or not 0 < tail <= full or not bool((values[:-1] == full).all()):
        return original(scores, costs, budget, block_ids)
    ids = torch.arange(len(costs), device=scores.device) if block_ids is None else block_ids
    canonical = torch.argsort(ids, stable=True)
    order = canonical[torch.argsort(scores[:, canonical], descending=True, stable=True)]
    if tail == full:
        retained = torch.arange(len(costs), device=scores.device)[None] < max(0, int(budget)//full)
        chosen = retained.expand_as(order)
    else:
        is_tail = order == len(costs)-1
        tail_position = is_tail.long().argmax(-1)
        full_quota = max(0, int(budget)//full)
        take_tail = (int(budget) >= tail) & ((tail_position < full_quota) | (int(budget) % full >= tail))
        full_count = ((int(budget)-take_tail.long()*tail)//full).clamp_min(0)
        rank = torch.arange(len(costs), device=scores.device)[None]
        full_rank = rank-(rank > tail_position[:, None]).long()
        chosen = ((~is_tail) & (full_rank < full_count[:, None])) | (is_tail & take_tail[:, None])
    return torch.zeros_like(scores).scatter(1, order, chosen.to(scores.dtype))


def head(self,q,kh,vh,kc,vc,state,aux_state,keep,step,chosen,collect_aux,detach_router,mass_only,collect_mechanisms,layer,head):
    from ..attention import masks,attend
    from ..common import nmse
    micro=self.query_microbatch;scores=[];delta_norm=[]
    def score_function(q0,ck,cv,st):
        route=self.router.score(q0,ck,cv,st,keep,step,detach_router,mass_only)
        return route['scores'],route['delta'].detach().square().sum()
    # Query rows are independent. Only this computation batch changes.
    for offset in range(0,len(q),SCORE_BATCH):
        args=(q[offset:offset+SCORE_BATCH],kc,vc,state)
        result=checkpoint(score_function,*args,use_reentrant=False) if torch.is_grad_enabled() else score_function(*args)
        scores.append(result[0]);delta_norm.append(result[1])
    scores=torch.cat(scores)
    # Larger GEMMs may perturb FP32 scores by a few ULPs. Near a discrete
    # selection boundary, recalculate complete ORIGINAL 16-query groups before
    # selection. This preserves the original kernel shape where ranking is
    # numerically ambiguous instead of weakening the full-network audit limit.
    with torch.no_grad():
        costs=state['costs'];budget=math.floor(keep*int(costs.sum()))
        preview=page_select(scores.detach(),costs,budget)
        full=int(costs[0]);tail=int(costs[-1])
        full_page=costs==full
        selected=(preview>.5)&full_page[None]
        rejected=(preview<.5)&full_page[None]
        low=scores.detach().masked_fill(~selected,torch.inf).amin(-1)
        high=scores.detach().masked_fill(~rejected,-torch.inf).amax(-1)
        guard=GUARD_ABSOLUTE+GUARD_RELATIVE*scores.detach().abs().amax(-1)
        ambiguous=(low-high)<=guard
        # General keep ratios can make the tail compete for a full-page slot.
        # Conservatively retain the original batches for that cost case.
        if tail<full and tail>budget%full:
            ambiguous=torch.ones_like(ambiguous)
        padding=(-len(q))%micro
        groups=torch.nn.functional.pad(ambiguous,(0,padding)).reshape(-1,micro).any(-1).nonzero().flatten().tolist()
    all_groups=math.ceil(len(q)/micro)
    complete_fallback=len(groups)>all_groups//2
    if complete_fallback:
        groups=list(range(all_groups))
    if groups:
        corrected=[];positions=[]
        for group in groups:
            offset=group*micro;args=(q[offset:offset+micro],kc,vc,state)
            result=checkpoint(score_function,*args,use_reentrant=False) if torch.is_grad_enabled() else score_function(*args)
            corrected.append(result[0]);positions.extend(range(offset,min(offset+micro,len(q))))
        # With many ambiguous groups, discard the larger-batch graph entirely:
        # do not backpropagate through a redundant all-zero-gradient graph.
        scores=torch.cat(corrected) if complete_fallback else scores.index_copy(0,torch.tensor(positions,device=q.device),torch.cat(corrected))
    mask,hard,soft=masks(scores,state['costs'],keep)
    actual=(hard*state['costs']).sum(-1)
    diagnostic=torch.stack((actual.sum(),actual.new_tensor(len(q)),
        ((soft<1e-4)|(soft>1-1e-4)).float().sum(),soft.new_tensor(soft.numel()),
        torch.stack(delta_norm).sum(),scores.detach().square().sum()))
    la=q.new_zeros((),dtype=torch.float32);lc=la.clone();records=[];aux_count=0
    def auxiliary(q0,hk,hv,ck,cv,st,m):
        e=self.router.sketch.estimate(q0.detach(),st,ck.detach(),cv.detach() if self.router.kind=='V' else None,self.router.kind)
        if not detach_router:
            m=self.router(q0.detach(),ck.detach(),cv.detach(),st,keep,step,True,mass_only)['mask']
        with torch.no_grad():
            dense=attend(q0.detach(),hk.detach(),hv.detach(),ck.detach(),cv.detach(),torch.ones_like(m),self.router.sketch.page_size)
        pred=attend(q0.detach(),hk.detach(),hv.detach(),ck.detach(),cv.detach(),m,self.router.sketch.page_size)
        return nmse(pred,dense),self.router.sketch.calibration_loss(q0.detach(),hk.detach(),ck.detach(),st,e)
    # Keep original auxiliary and diagnostic sample granularity unchanged.
    for index,offset in enumerate(range(0,len(q),micro)):
        if not (collect_aux or collect_mechanisms) or index not in chosen:
            continue
        qq=q[offset:offset+micro];mm=mask[offset:offset+micro]
        if collect_aux:
            args=(qq,kh,vh,kc,vc,aux_state,mm)
            aa,cc=checkpoint(auxiliary,*args,use_reentrant=False) if torch.is_grad_enabled() else auxiliary(*args)
            la=la+aa;lc=lc+cc;aux_count+=1
        if collect_mechanisms:
            from ..diagnostics import mechanisms
            route=self.router.score(qq,kc,vc,state,keep,step,detach_router,mass_only)
            route.update(hard=hard[offset:offset+micro],soft=soft[offset:offset+micro],mask=mm)
            record=mechanisms(self.router,qq.detach(),kh.detach(),vh.detach(),kc.detach(),vc.detach(),state,route,keep)
            records.append(dict(layer=layer,head=head,query_offset=offset,query_count=len(qq),**record))
    return mask,la,lc,diagnostic,aux_count,records


def multi_backward(ctx,grad):
    from ..attention import _native_multi,_weights
    q,kh,vh,kc,vc,mask=ctx.saved_tensors;size=ctx.size
    with torch.enable_grad():
        inputs=[x.detach().requires_grad_() for x in (q,kh,vh,kc,vc)]
        output=_native_multi(*inputs,mask.detach(),size)
        raw_grad=torch.autograd.grad(output,inputs,grad_outputs=grad,allow_unused=False)
    # Frozen router needs no continuous gate VJP; native QKV path still runs.
    if not ctx.needs_input_grad[5]:
        return *raw_grad,None,None
    mask_grad=[]
    for h in range(q.shape[1]):
        key=torch.cat((kh[:,h],kc[:,h]));value=torch.cat((vh[:,h],vc[:,h]))
        value_double=value.double();pieces=[]
        for start in range(0,len(q),GATE_BATCH):
            sl=slice(start,start+GATE_BATCH)
            out,_,ungated=_weights(q[sl,h],key,value,mask[h,sl],len(kh),size)
            g=grad[sl,h].double();dot=g@value_double.T-(g*out).sum(-1,keepdim=True)
            token=ungated[:,:len(kh)]*dot[:,:len(kh)]
            pieces.append(torch.nn.functional.pad(token,(0,mask.shape[-1]*size-len(kh))).reshape(len(q[sl]),-1,size).sum(-1))
        mask_grad.append(torch.cat(pieces))
    dm=torch.stack(mask_grad).to(mask.dtype)
    if not all(torch.isfinite(x).all() for x in (*raw_grad,dm)):
        raise FloatingPointError('Multi-head native gradient')
    return *raw_grad,dm,None


def install():
    from .. import attention,backbone
    if _originals:
        return
    _originals.update(select=attention.select,head=backbone.FunctionalBackbone._head,
                      backward=attention.RawMultiAttention.backward)
    attention.select=page_select
    backbone.FunctionalBackbone._head=head
    attention.RawMultiAttention.backward=staticmethod(multi_backward)


def uninstall():
    from .. import attention,backbone
    if not _originals:
        return
    attention.select=_originals['select']
    backbone.FunctionalBackbone._head=_originals['head']
    attention.RawMultiAttention.backward=staticmethod(_originals['backward'])
    _originals.clear()
