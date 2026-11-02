"""Small budgeted models: page-local KV pooling, mass routing, contextual one swap."""
import math
import torch
from torch import nn

NF=8;NR=14

def ste_bf16(x):return x+(x.to(torch.bfloat16).float()-x).detach()
class Layer(nn.Module):
    def __init__(self,old):
        super().__init__()
        self.basis=nn.Parameter(old['basis'].to(torch.bfloat16).float().clone())
        self.center=nn.Parameter(old['center'].float().clone())
        self.uk=nn.Parameter(torch.zeros(12,128));self.uv=nn.Parameter(torch.zeros(12,128))
        self.route=nn.Parameter(torch.zeros(12,NF));self.swap=nn.Parameter(torch.zeros(12,NR))
    def value(self,k,v):
        # h,p,64,d; assignments remain fixed within each physical page.
        a=torch.einsum('hptd,hd->hpt',k,self.uk)+torch.einsum('hptd,hd->hpt',v,self.uv)
        a=(2*torch.tanh((a-a.mean(-1,keepdim=True))/2)).softmax(-1)
        vm=torch.einsum('hpt,hptd->hpd',a,v)
        basis=ste_bf16(self.basis);code=ste_bf16((vm-self.center[:,None])@basis)
        return code,self.center[:,None]+code@basis.transpose(1,2)
    @torch.no_grad()
    def project(self):
        self.uk.clamp_(-.5,.5);self.uv.clamp_(-.5,.5)
        self.basis.copy_(torch.linalg.qr(self.basis,mode='reduced')[0])
        self.route.clamp_(-5,5);self.swap.clamp_(-5,5)

def probe_mass(q,pk):
    # q: h,t,4,d, pk: h,p,4,d. Four real queries normalized independently.
    h,t,m,d=q.shape;p=pk.shape[1]
    logits=(q.reshape(h,t*m,d)@pk.flatten(1,2).transpose(1,2)/math.sqrt(d)).reshape(h,t,m,p,4)
    logz=logits.logsumexp(-1)+math.log(16)
    prob=logz.softmax(-1)
    return logz,prob,prob.mean(2)

def features(prob,pk,vvar):
    h,t,m,p=prob.shape;mass=prob.mean(2);lm=mass.clamp_min(1e-12).log()
    lp=prob.clamp_min(1e-12).log()
    peak=(prob.max(2).values.clamp_min(1e-12).log()-lm).clamp(0,10)/5
    spread=lp.std(2,unbiased=False).clamp(0,5)/5
    kvar=(pk-pk.mean(2,keepdim=True)).square().mean((2,3)).log1p()
    current=(torch.arange(p,device=pk.device)>=p-72).float()[None,None].expand(h,t,-1)
    vv=vvar.float().clamp_min(0).log1p()[:,None].expand(-1,t,-1)
    kv=kvar[:,None].expand(-1,t,-1);lc=lm.clamp(-20,0)/5
    return torch.stack((torch.ones_like(lm),lc,peak,spread,vv,kv,current,current*lc),-1)

def pick(score,keep,own):
    score=score.clone();score.scatter_(-1,own[...,None],float('inf'))
    mask=torch.zeros_like(score,dtype=torch.bool);mask.scatter_(-1,score.topk(keep,-1).indices,True)
    return mask

def route_score(mass,feat,weight):return mass.clamp_min(1e-12).log()+torch.einsum('htpf,hf->htp',feat,weight)

def swap_features(feat,mass,code,mask):
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=weighted@code/den;allvalue=mass@code
    missing=allvalue-selected;diff=code[:,None]-selected[:,:,None]
    energy=selected.square().sum(-1,keepdim=True).clamp_min(.01)
    align=(diff*missing[:,:,None]).sum(-1)/energy
    mag=diff.square().sum(-1)/energy
    extra=torch.stack((mask.float(),(mass/den).clamp(0,2),align.clamp(-10,10)/5,mag.clamp(0,10)/5,(mass*align).clamp(-2,2),(mass*mag).clamp(0,2)),-1)
    return torch.cat((feat,extra),-1)

def correct(mask,score,own,threshold=0.):
    rem=score.masked_fill(~mask,float('inf')).clone();rem.scatter_(-1,own[...,None],float('inf'))
    low,drop=rem.min(-1);high,add=score.masked_fill(mask,-float('inf')).max(-1)
    change=high>low+threshold;out=mask.clone()
    out.scatter_(-1,drop[...,None],~change[...,None]);out.scatter_(-1,add[...,None],change[...,None])
    return out,change

@torch.no_grad()
def event(rt,q,k,v,lut):
    h,d=12,128;p=rt.active_pk.shape[1];device=q.device
    tiles=torch.tensor([0,23,47,71],device=device);offsets=torch.tensor([0,21,42,63],device=device)
    ids=(tiles[:,None]*64+offsets).flatten()
    query=q[0].transpose(0,1)[:,ids].float().reshape(h,4,4,d)
    kp=k[0].transpose(0,1).float().reshape(h,p,64,d);vp=v[0].transpose(0,1).float().reshape_as(kp)
    logits=torch.einsum('htqd,hpnd->htqpn',query,kp)/math.sqrt(d)
    logz=logits.logsumexp(-1);values=torch.einsum('htqpn,hpnd->htqpd',logits.softmax(-1),vp)
    truth=logz.softmax(-1);target=(truth[...,None]*values).sum(-2)
    pk=rt.active_pk.detach();az,prob,mass=probe_mass(query,pk)
    approx=(az-logz.logsumexp(-1,keepdim=True)).exp()
    vm=vp.mean(2);vvar=(vp-vm[:,:,None]).square().mean((2,3)).to(torch.bfloat16).float()
    feat=features(prob,pk,vvar)
    mask=torch.zeros(h,4,p,device=device,dtype=torch.bool).scatter_(-1,lut[0,:,tiles].long(),True)
    return dict(k=kp,v=vp,truth=truth,values=values,target=target,approx=approx,mass=mass,feat=feat,source_mask=mask,own=(tiles+p-72)[None].expand(h,-1),keep=lut.shape[-1],old_code=rt.active_codes.float().detach())

def mixed(e,av,mask):
    mask=mask[:,:,None];w=e['truth']*mask;a=e['approx']*(~mask)
    N=(w[...,None]*e['values']).sum(-2)+torch.einsum('htqp,hpd->htqd',a,av)
    Z=w.sum(-1)+a.sum(-1)
    return N/Z.clamp_min(1e-12)[...,None],N,Z

def nmse(y,e):return ((y-e['target']).square().sum(-1)/e['target'].square().sum(-1).clamp_min(1e-8)).mean()

@torch.no_grad()
def gain_targets(e,av,mask):
    y,N,Z=mixed(e,av,mask);dw=e['truth']-e['approx']
    dn=e['truth'][...,None]*e['values']-e['approx'][...,None]*av[:,None,None]
    sign=(1-2*mask.float())[:,:,None]
    changed=(N[:,:,:,None]+sign[...,None]*dn)/(Z[:,:,:,None]+sign*dw).clamp_min(1e-12)[...,None]
    en=e['target'].square().sum(-1).clamp_min(1e-8)
    before=(y-e['target']).square().sum(-1)/en
    after=(changed-e['target'][:,:,:,None]).square().sum(-1)/en[:,:,:,None]
    gains=((before[...,None]-after)*sign).mean(2)
    scale=gains.std(-1,keepdim=True,unbiased=False).clamp_min(1e-5)
    return (gains/scale).clamp(-4,4)

@torch.no_grad()
def evaluate(layer,e,oldbasis,oldcenter):
    code,av=layer.value(e['k'],e['v']);oldv=oldcenter[:,None]+e['old_code']@oldbasis.float().transpose(1,2)
    base=pick(e['mass'],e['keep'],e['own']);learned=pick(route_score(e['mass'],e['feat'],layer.route),e['keep'],e['own'])
    sf=swap_features(e['feat'],e['mass'],code,learned);sm=torch.einsum('htpf,hf->htp',sf,layer.swap)
    refined,changed=correct(learned,sm,e['own'])
    rows={}
    for name,m,v in [('baseline',base,oldv),('route',learned,oldv),('value',base,av),('joint',learned,av),('swap',refined,av)]:
        rows[name]=float(nmse(mixed(e,v,m)[0],e))
    return dict(nmse=rows,swap_fraction=float(changed.float().mean()))
