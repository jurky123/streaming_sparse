"""Teacher-only exact page statistics and stable original-KV mask output."""
from dataclasses import dataclass
import math
import torch


def pages(k,v,size=64):
    if k.ndim!=2 or k.shape!=v.shape or k.shape[-1]!=128 or not len(k):raise ValueError('Invalid128-d KV')
    b=math.ceil(len(k)/size);pad=b*size-len(k)
    kk=torch.nn.functional.pad(k.float(),(0,0,0,pad)).reshape(b,size,128)
    vv=torch.nn.functional.pad(v.float(),(0,0,0,pad)).reshape(b,size,128)
    valid=torch.arange(b*size,device=k.device).reshape(b,size)<len(k)
    return kk,vv,valid


def page_costs(n,size,device):
    return (n-torch.arange(math.ceil(n/size),device=device)*size).clamp(0,size).float()


def partition_hash(indices,size):
    import hashlib
    return hashlib.sha256(str(size).encode()+indices.detach().long().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def current_stats(q,k,v,visible=None):
    if not len(k):raise ValueError('Current visible KV must be nonempty')
    logits=q.float()@k.float().T/math.sqrt(128)
    if visible is not None:
        if visible.shape!=logits.shape or not visible.any(-1).all():raise ValueError('Bad current visibility')
        logits=logits.masked_fill(~visible,-torch.inf)
    return torch.logsumexp(logits,-1),logits.softmax(-1)@v.float()


class MaskOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx,mask,L,mu,Lc,oc):
        if mask.shape!=L.shape or (mask<0).any() or not torch.isfinite(mask).all():raise ValueError('Invalid nonnegative per-query mask')
        logm=mask.clamp_min(1e-30).log().masked_fill(mask==0,-torch.inf)
        logs=torch.cat((L+logm,Lc[:,None]),-1)
        total=torch.logsumexp(logs,-1)
        weights=logs.softmax(-1)
        out=(weights[:,:-1,None]*mu).sum(1)+weights[:,-1,None]*oc
        ctx.save_for_backward(L,mu,total,out)
        return out

    @staticmethod
    def backward(ctx,grad):
        L,mu,total,out=ctx.saved_tensors
        # Only teacher constants feed this operator. The derivative is exact when
        # L-logZselected<=30; bounded FP32 proxy in pathological extreme tests.
        derivative=(L-total[:,None]).clamp(max=30).exp()*((mu-out[:,None])*grad[:,None]).sum(-1)
        return derivative,None,None,None,None


@dataclass
class TeacherStats:
    logz:torch.Tensor
    mu:torch.Tensor
    current_logz:torch.Tensor
    current_output:torch.Tensor

    def output(self,mask):return MaskOutput.apply(mask.float(),self.logz,self.mu,self.current_logz,self.current_output)
    def dense(self):return self.output(torch.ones_like(self.logz))
    def geometry(self):return geometry(self.logz,self.mu,self.current_logz,self.current_output)
    def gradient_guard_count(self,mask):
        lm=mask.clamp_min(1e-30).log().masked_fill(mask==0,-torch.inf)
        total=torch.logsumexp(torch.cat((self.logz+lm,self.current_logz[:,None]),-1),-1)
        return int(((self.logz-total[:,None])>30).sum())


def geometry(logz,mu,Lc,oc,history_only=False):
    # Bulk QK/prototype operations stay FP32. Use FP64 only for normalized
    # geometry and short reductions: FP32 omitted residuals can cancel almost
    # completely, and division by a small retained mass magnifies that error.
    logs=logz.double();values=mu.double();local=oc.double()
    if history_only:
        total=torch.logsumexp(logs,-1);mass=logs.softmax(-1)
        current_mass=torch.zeros_like(total)
        dense=(mass[...,None]*values).sum(1)
    else:
        combined=torch.cat((logs,Lc.double()[:,None]),-1)
        total=torch.logsumexp(combined,-1);weights=combined.softmax(-1)
        mass=weights[:,:-1];current_mass=weights[:,-1]
        dense=(mass[...,None]*values).sum(1)+current_mass[:,None]*local
    residual=mass[...,None]*(values-dense[:,None]);dtype=logz.dtype
    return dict(logz=logz,mu=mu,total_logz=total.to(dtype),mass=mass.to(dtype),
                current_mass=current_mass.to(dtype),dense=dense.to(dtype),d=residual.to(dtype),
                current_logz=Lc,current_output=oc,_mass64=mass,_current_mass64=current_mass,
                _dense64=dense,_d64=residual)


def exclusive_sum(value):
    """Sum all other blocks without subtracting a dominant block from a total."""
    zero=torch.zeros_like(value[:,:1])
    left=torch.cat((zero,value[:,:-1].cumsum(1)),1)
    right=torch.cat((value[:,1:].flip((1,)).cumsum(1).flip((1,)),zero),1)
    return left+right


def retained_residual(g,mask):
    """D = z*estimated_dense - retained_numerator, stable even for tiny z.

    All inputs are predicted geometry for deployment; no teacher-only input is
    introduced. This is exactly the omitted-residual identity in real arithmetic.
    """
    selected=mask.double()*g['_mass64']
    z=g['_current_mass64']+selected.sum(-1)
    numerator=(g['_current_mass64'][:,None]*g['current_output'].double()+
               (selected[...,None]*g['mu'].double()).sum(1))
    return z[:,None]*g['_dense64']-numerator,z


@torch.no_grad()
def teacher(q,event,size=64,permutation=None):
    kh,vh=event['k_hist'],event['v_hist']
    if permutation is not None:kh,vh=kh[permutation],vh[permutation]
    k,v,valid=pages(kh,vh,size)
    logits=torch.einsum('qd,btd->qbt',q.float(),k)/math.sqrt(128)
    logits=logits.masked_fill(~valid[None],-torch.inf)
    L=torch.logsumexp(logits,-1)
    mu=torch.einsum('qbt,btd->qbd',logits.softmax(-1),v)
    Lc,oc=current_stats(q,event['k_current'],event['v_current'],event.get('current_visible'))
    return TeacherStats(L,mu,Lc,oc)


def singleton(g):
    denominator=g['_current_mass64'][:,None]+exclusive_sum(g['_mass64'])
    return (g['_d64']/denominator.clamp_min(1e-12)[...,None]).square().sum(-1).to(g['mass'].dtype)


def set_gain(g,initial):
    mass=g['_mass64'];mu=g['mu'].double();dense=g['_dense64'];selected=initial.double()*mass
    local_num=g['_current_mass64'][:,None]*g['current_output'].double()
    selected_num=selected[...,None]*mu
    D,z=retained_residual(g,initial);z=z.clamp_min(1e-12)
    e0=(D/z[:,None]).square().sum(-1)
    numerator=local_num+selected_num.sum(1)
    za=(z[:,None]+mass).clamp_min(1e-12)
    added=((numerator[:,None]+mass[...,None]*mu)/za[...,None]-dense[:,None]).square().sum(-1)
    # The (M_b-1) term is zero for selected blocks in the hard forward, but
    # retains the original z-m_b / D+d_b surrogate derivative through M0.
    remainder=(initial.double()-1)*mass
    zr=(g['_current_mass64'][:,None]+exclusive_sum(selected)+remainder).clamp_min(1e-12)
    nr=local_num[:,None]+exclusive_sum(selected_num)+remainder[...,None]*mu
    removed=(nr/zr[...,None]-dense[:,None]).square().sum(-1)
    gains=torch.where(initial.detach()>.5,removed-e0[:,None],e0[:,None]-added)
    u=torch.asinh(gains/(e0[:,None]+1e-12))
    return {k:v.to(g['mass'].dtype) for k,v in dict(D=D,z=z,error=e0,gain=gains,u=u).items()}


def interaction(g,mask):
    deleted=1-mask
    d=g['_d64']*deleted.double()[...,None]
    diag=d.square().sum((1,2));total=d.sum(1).square().sum(-1)
    gram=g['_d64']@g['_d64'].transpose(-1,-2)
    eligible=torch.triu((deleted[:,:,None]>.5)&(deleted[:,None,:]>.5),diagonal=1)
    negative=((gram<0)&eligible).sum((1,2))/eligible.sum((1,2)).clamp_min(1)
    return dict(kappa=(total-diag)/(diag+1e-12),interaction=total-diag,diagonal=diag,
                negative_pair_fraction=negative,removed_mass=(deleted*g['mass']).sum(-1))
