"""Exact hard subset attention with ONE raw-QKV path and a continuous gate VJP.

Unlike v07's teacher-only operator this differentiates the *executing* QKV.
No log(mask), masked_fill of a zero gate, straight-through QKV duplication or
clipped gate derivative. Saved inputs are recomputed once in backward.
"""
import math
import torch
from ..v05.primitives import BudgetGate, select


def costs(n, size=64, device=None):
    return (n-torch.arange(math.ceil(n/size), device=device)*size).clamp(0,size).float()


def masks(scores, page_costs, keep, temperature=.2):
    budget = math.floor(float(keep)*int(page_costs.sum()))
    hard = select(scores, page_costs, budget)
    soft = BudgetGate.apply(scores, page_costs, budget, temperature)
    return hard+(soft-soft.detach()), hard, soft


def _weights(q,k,v,mask,history,size):
    work = torch.float64 if q.dtype == torch.float64 else torch.float32
    logits = q.to(work)@k.to(work).T/math.sqrt(q.shape[-1])
    # Detached common shift changes no mathematical derivative. FP64 keeps small
    # retained/current mass representable even when a removed page dominates.
    shift = logits.detach().amax(-1,keepdim=True)
    exp = (logits.double()-shift.double()).exp()
    token_mask = torch.cat((mask.repeat_interleave(size,-1)[:,:history],
                           mask.new_ones((len(q),len(k)-history))),-1).double()
    weighted = exp*token_mask
    denominator = weighted.sum(-1,keepdim=True)
    if (denominator<=0).any() or not torch.isfinite(denominator).all():
        raise FloatingPointError('Unrepresentable retained attention normalizer')
    weights = weighted/denominator
    result = weights@v.double()
    return result,weights,exp/denominator


def _native(q,kh,vh,kc,vc,mask,size):
    k=torch.cat((kh,kc));v=torch.cat((vh,vc))
    visible=torch.cat((mask.repeat_interleave(size,-1)[:,:len(kh)]>.5,
                       torch.ones(len(q),len(kc),device=q.device,dtype=torch.bool)),-1)
    # Full native query length, dtype and backend, including the all-keep case.
    # Hard output does not pretend the boolean mask itself supplies gradients.
    attention_mask=None if bool(visible.all()) else visible[None,None]
    return torch.nn.functional.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None],
        attn_mask=attention_mask,dropout_p=0.,is_causal=False)[0,0]


class RawAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,kh,vh,kc,vc,mask,size):
        if q.ndim!=2 or kh.shape!=vh.shape or kc.shape!=vc.shape or len(kc)==0:
            raise ValueError('Expected single-head Q/history/current tensors, nonempty current')
        if mask.shape!=(len(q),math.ceil(len(kh)/size)) or (mask<0).any():
            raise ValueError('Expected nonnegative per-query page mask')
        ctx.native=q.dtype in (torch.bfloat16,torch.float16) and bool(((mask==0)|(mask==1)).all())
        if ctx.native:result=_native(q,kh,vh,kc,vc,mask,size)
        else:
            k=torch.cat((kh,kc));v=torch.cat((vh,vc))
            result,_,_=_weights(q,k,v,mask,len(kh),size)
        ctx.save_for_backward(q,kh,vh,kc,vc,mask)
        ctx.size=size
        return result.to(q.dtype)

    @staticmethod
    def backward(ctx,grad):
        q,kh,vh,kc,vc,mask=ctx.saved_tensors;size=ctx.size
        k=torch.cat((kh,kc));v=torch.cat((vh,vc));n=len(kh)
        if ctx.native:
            # ONE executing QKV derivative path, through the same native hard
            # attention. The independently supplied gate VJP is continuous;
            # it must not also add a second QKV derivative.
            with torch.enable_grad():
                inputs=[x.detach().requires_grad_() for x in (q,kh,vh,kc,vc)]
                result=_native(*inputs,mask.detach(),size)
                raw_grad=torch.autograd.grad(result,inputs,grad_outputs=grad,allow_unused=False)
            parts=[]
            for start in range(0,len(q),16):
                sl=slice(start,start+16)
                out,_,ungated=_weights(q[sl],k,v,mask[sl],n,size)
                g=grad[sl].double()
                dot=g@v.double().T-(g*out).sum(-1,keepdim=True)
                token=ungated[:,:n]*dot[:,:n]
                parts.append(torch.nn.functional.pad(token,(0,mask.shape[-1]*size-n)).reshape(len(q[sl]),-1,size).sum(-1))
            dm=torch.cat(parts).to(mask.dtype)
            if not all(torch.isfinite(x).all() for x in (*raw_grad,dm)):raise FloatingPointError('Native QKV/gate gradient is nonfinite')
            return *raw_grad,dm,None
        out,p,ungated=_weights(q,k,v,mask,n,size)
        g=grad.double()
        value_dot=g@v.double().T-(g*out).sum(-1,keepdim=True)
        logits_grad=p*value_dot
        work=torch.float64 if q.dtype==torch.float64 else torch.float32
        dl=logits_grad.to(work)
        dq=dl@k.to(work)/math.sqrt(q.shape[-1])
        dk=dl.T@q.to(work)/math.sqrt(q.shape[-1])
        dv=p.to(work).T@grad.to(work)
        token_gate=ungated[:,:n]*value_dot[:,:n]
        pad=mask.shape[-1]*size-n
        dm=torch.nn.functional.pad(token_gate,(0,pad)).reshape(len(q),-1,size).sum(-1)
        values=(dq,dk[:n],dv[:n],dk[n:],dv[n:],dm)
        if not all(torch.isfinite(x).all() for x in values):
            raise FloatingPointError('Nonfinite raw attention/continuous gate gradient')
        return *(x.to(t.dtype) for x,t in zip(values,(q,kh,vh,kc,vc,mask))),None


def attend(q,kh,vh,kc,vc,mask,size=64):
    # custom Function math must not inherit a BF16 autocast from the backbone.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return RawAttention.apply(q,kh,vh,kc,vc,mask,size)


def reference(q,kh,vh,kc,vc,mask,size=64):
    """Independent direct per-query gather oracle, never used to score/train."""
    out=[]
    for i in range(len(q)):
        ix=torch.where(mask[i].repeat_interleave(size)[:len(kh)]>.5)[0]
        k=torch.cat((kh[ix],kc));v=torch.cat((vh[ix],vc))
        work=torch.float64 if q.dtype==torch.float64 else torch.float32
        out.append((q[i:i+1].to(work)@k.to(work).T/math.sqrt(q.shape[-1])).softmax(-1)@v.to(work))
    return torch.cat(out)


def block_logz(q,kh,size=64):
    n=len(kh);b=math.ceil(n/size)
    work=torch.float64 if q.dtype==torch.float64 else torch.float32
    k=torch.nn.functional.pad(kh.to(work),(0,0,0,b*size-n)).reshape(b,size,-1)
    l=torch.einsum('qd,btd->qbt',q.to(work),k)/math.sqrt(q.shape[-1])
    valid=torch.arange(b*size,device=q.device).reshape(b,size)<n
    return l.masked_fill(~valid[None],-torch.inf).logsumexp(-1)


def _native_multi(q,kh,vh,kc,vc,mask,size):
    key=torch.cat((kh,kc));value=torch.cat((vh,vc))
    visible=None
    if not bool((mask==1).all()):
        historical=(mask>.5).repeat_interleave(size,-1)[...,:len(kh)]
        current=torch.ones(mask.shape[0],mask.shape[1],len(kc),device=q.device,dtype=torch.bool)
        visible=torch.cat((historical,current),-1)[None]
    result=torch.nn.functional.scaled_dot_product_attention(q.permute(1,0,2)[None],key.permute(1,0,2)[None],
        value.permute(1,0,2)[None],attn_mask=visible,dropout_p=0.,is_causal=False)
    return result[0].permute(1,0,2).contiguous()


class RawMultiAttention(torch.autograd.Function):
    """Preserve the original12-head native backend's arithmetic, not just dtype."""
    @staticmethod
    def forward(ctx,q,kh,vh,kc,vc,mask,size):
        ctx.save_for_backward(q,kh,vh,kc,vc,mask);ctx.size=size
        return _native_multi(q,kh,vh,kc,vc,mask,size)
    @staticmethod
    def backward(ctx,grad):
        q,kh,vh,kc,vc,mask=ctx.saved_tensors;size=ctx.size
        with torch.enable_grad():
            inputs=[x.detach().requires_grad_() for x in (q,kh,vh,kc,vc)]
            output=_native_multi(*inputs,mask.detach(),size)
            raw_grad=torch.autograd.grad(output,inputs,grad_outputs=grad,allow_unused=False)
        mask_grad=[]
        for head in range(q.shape[1]):
            key=torch.cat((kh[:,head],kc[:,head]));value=torch.cat((vh[:,head],vc[:,head]))
            pieces=[]
            for start in range(0,len(q),16):
                sl=slice(start,start+16)
                out,_,ungated=_weights(q[sl,head],key,value,mask[head,sl],len(kh),size)
                g=grad[sl,head].double();dot=g@value.double().T-(g*out).sum(-1,keepdim=True)
                token=ungated[:,:len(kh)]*dot[:,:len(kh)]
                pieces.append(torch.nn.functional.pad(token,(0,mask.shape[-1]*size-len(kh))).reshape(len(q[sl]),-1,size).sum(-1))
            mask_grad.append(torch.cat(pieces))
        dm=torch.stack(mask_grad).to(mask.dtype)
        if not all(torch.isfinite(x).all() for x in (*raw_grad,dm)):raise FloatingPointError('Multi-head native gradient')
        return *raw_grad,dm,None


def attend_multi(q,kh,vh,kc,vc,mask,size=64):
    with torch.autocast(device_type=q.device.type,enabled=False):
        return RawMultiAttention.apply(q,kh,vh,kc,vc,mask,size)
