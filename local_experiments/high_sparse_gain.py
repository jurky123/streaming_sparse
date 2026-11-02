"""Fixed-budget pairwise swap targets relative to omitted-page compensation."""
import torch
from .high_sparse_learn import pick,swap_features,mixed,nmse,correct,gain_targets

def inputs(e):
    mask=pick(e['mass'],e['keep'],e['own'])
    return mask,swap_features(e['feat'],e['mass'],e['old_code'],mask)
def score(mass,feat,w,strength=1.):
    return mass.clamp_min(1e-12).log()+strength*torch.tanh(torch.einsum('htpf,hf->htp',feat,w))

@torch.no_grad()
def targets(e,av,mask,n=8):
    # Use cheap exact one-page deltas only to shortlist. Labels use simultaneous
    # add/remove with a COMMON updated normalizer, never independent gain sums.
    g=gain_targets(e,av,mask)
    remove=g.masked_fill(~mask,float('inf'));remove.scatter_(-1,e['own'][...,None],float('inf'))
    add=g.masked_fill(mask,-float('inf')).topk(n,-1).indices
    drop=remove.topk(n,-1,largest=False).indices
    y,N,Z=mixed(e,av,mask)
    dw=e['truth']-e['approx'];dn=e['truth'][...,None]*e['values']-e['approx'][...,None]*av[:,None,None]
    def take(x,ids):
        if x.ndim==5:return x.gather(3,ids[:,:,None,:,None].expand(-1,-1,x.shape[2],-1,x.shape[-1]))
        return x.gather(3,ids[:,:,None,:].expand(-1,-1,x.shape[2],-1))
    NN=N[:,:,:,None,None,:]+take(dn,add)[:,:,:,:,None,:]-take(dn,drop)[:,:,:,None,:,:]
    ZZ=Z[:,:,:,None,None]+take(dw,add)[:,:,:,:,None]-take(dw,drop)[:,:,:,None,:]
    changed=NN/ZZ.clamp_min(1e-12)[...,None]
    norm=e['target'].square().sum(-1).clamp_min(1e-8)
    before=((y-e['target']).square().sum(-1)/norm).mean(2)
    after=((changed-e['target'][:,:,:,None,None]).square().sum(-1)/norm[:,:,:,None,None]).mean(2)
    benefit=before[:,:,None,None]-after
    return add,drop,benefit,before

def loss(e,feat,w,add,drop,gain,before):
    s=score(e['mass'],feat,w)
    margin=s.gather(-1,add)[...,None]-s.gather(-1,drop)[:,:,None,:]
    weight=(gain.abs()/before[:,:,None,None].clamp_min(1e-6)).clamp(max=.5)
    target=gain.sign()
    rank=(torch.nn.functional.softplus(-target*margin)*weight).sum()/weight.sum().clamp_min(1e-8)
    return rank+1e-3*w.square().mean()

@torch.no_grad()
def evaluate(e,av,w):
    mask,feat=inputs(e);out={}
    for strength in [0.,.25,.5,1.]:
        new,change=correct(mask,score(e['mass'],feat,w,strength),e['own']) if strength else (mask,torch.zeros_like(e['own'],dtype=torch.bool))
        out[str(strength)]=dict(nmse=float(nmse(mixed(e,av,new)[0],e)),swap_fraction=float(change.float().mean()))
    return out
