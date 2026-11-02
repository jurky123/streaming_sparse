"""Page-local soft mixture and hard-forward, biased gate-backward training."""
import math
import torch
from torch import nn
from .learned_pool import Pool, Bank
from .learned_cache.model import initial_mask, output_for_masks


class PageMixture(nn.Module):
    """core.part semantics: token->prototype softmax, shared K/V, dynamic mass."""
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(30,12,4,128)*.01)
        self.bias=nn.Parameter(torch.zeros(30,12,4))

    def assignment(self,h):
        k=h['k'].reshape(3,-1,64,128)
        logits=torch.einsum('hbtd,hrd->hbtr',k,self.weight[h['layer'],h['heads']])+self.bias[h['layer'],h['heads']][:,None,None]
        return logits.softmax(-1)

    def forward(self,h):
        a=self.assignment(h);count=a.sum(2)
        k=h['k'].reshape(3,-1,64,128);v=h['v'].reshape_as(k)
        return (torch.einsum('hbtr,hbtd->hbrd',a,k)/count.clamp_min(1e-20)[...,None],
                torch.einsum('hbtr,hbtd->hbrd',a,v)/count.clamp_min(1e-20)[...,None],count)


class PageGate(torch.autograd.Function):
    """Exact selected-set output forward; gate Jacobian is a biased surrogate.

    Use a per-query stabilizer from selected pages only. Clamp backward scale
    for omitted pages to exp(30); forward never depends on the clamped values.
    No claim of an exact derivative of discrete top-k.
    """
    @staticmethod
    def forward(ctx,gate,hard,z,values):
        out=output_for_masks(z,values,hard[:,:,None])[:,:,0]
        selected_logz=z.masked_fill(~hard,-torch.inf).logsumexp(-1)
        scale=(z-selected_logz[:,:,None]).clamp(max=30).exp()
        ctx.save_for_backward(scale,values,out)
        return out

    @staticmethod
    def backward(ctx,grad):
        scale,values,out=ctx.saved_tensors
        gate_grad=scale*((values-out[:,:,None])*grad[:,:,None]).sum(-1)
        return gate_grad,None,None,None


def hard_output(predicted_z,e,temperature=.5):
    hard=initial_mask(predicted_z)
    boundary_idx=predicted_z[...,:-1].masked_fill(~hard[...,:-1],torch.inf).argmin(-1,keepdim=True)
    boundary=predicted_z.gather(-1,boundary_idx)
    soft=((predicted_z[...,:-1]-boundary)/temperature).sigmoid()
    gate=torch.cat((soft,torch.ones_like(soft[...,:1])),-1)
    gate=hard.to(gate.dtype)+(gate-gate.detach())
    return PageGate.apply(gate,hard,e['z'],e['values'])


def make_model(representation):
    return PageMixture() if representation=='mixture' else Pool()


def prototypes(model,h,representation):
    return model(h) if representation=='mixture' else model(h,True)
