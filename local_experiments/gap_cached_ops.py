import math,torch
from .high_sparse_learn import features,correct
from .high_sparse_gain import score
def cached_pool(self,k,v,cheap=False):
    h,p,_,d=k.shape;k=k.float();v=v.float()
    if cheap:
        kc=k-k.mean(-2,keepdim=True)
        f=torch.einsum('hptd,hrd->hptr',kc,self.bank)/128**.5
        f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    else:f=self.group_features(k,v)
    f=f.reshape(h,p,4,16,32);kg=k.reshape(h,p,4,16,d);vg=v.reshape_as(kg)
    z=torch.einsum('hprtd,hrd->hprt',kg-kg.mean(-2,keepdim=True),self.pool_k)
    z=z+torch.einsum('hprtd,hrd->hprt',vg-vg.mean(-2,keepdim=True),self.pool_v)+torch.einsum('hprtf,hrf->hprt',f,self.pool_feature)
    z=z-z.mean(-1,keepdim=True);a=(2*torch.tanh(z/2)).softmax(-1)
    pk=(a[...,None]*kg).sum(-2);pv=(a[...,None]*vg).sum(-2)
    # Preserve page-level K mean and encoded V mean. No new cache bytes.
    pk=pk-pk.mean(2,keepdim=True)+k.mean(2,keepdim=True)
    basis=self.basis.to(torch.bfloat16).float()
    codes=torch.einsum('hprd,hdk->hprk',pv-self.center[:,None,None],basis)
    base=torch.einsum('hpd,hdk->hpk',v.mean(2)-self.center[:,None],basis).to(torch.bfloat16)
    haar=self._cached_haar
    detail=(torch.einsum('kr,hprd->hpkd',haar,codes)/4).to(torch.bfloat16)
    return pk.contiguous(),base.contiguous(),detail.contiguous()
def cached_route(m,q,pk,base,detail,cov,vvar,offsets,training=False):
    h,p,_,d=pk.shape;t=q.shape[1]//64;n=offsets.numel();keep=int(math.floor(p*.1+1e-6))
    query=q[0].transpose(0,1).float().reshape(h,t,64,d).index_select(2,offsets)
    logits=(query.reshape(h,t*n,d)@pk.flatten(1,2).transpose(1,2)/math.sqrt(d)).reshape(h,t,n,p,4)
    prob=(logits.logsumexp(-1)+math.log(16)).softmax(-1);mass=prob.mean(2);feat=features(prob,pk,vvar)
    within=logits.softmax(-1);weight=(prob[...,None]*within).mean(2)/mass[...,None].clamp_min(1e-12)
    haar=m._cached_haar
    coefficients=torch.einsum('htpr,ar->htpa',weight,haar)
    conditional=base.float()[:,None]+torch.einsum('htpa,hpad->htpd',coefficients,detail.float())
    # First-order diagonal projected KV-covariance proxy, not an exact page oracle.
    qp=query@m.basis.to(torch.bfloat16).float()[:,None]/math.sqrt(d)
    qbar=torch.einsum('htmp,htmd->htpd',prob,qp)/(n*mass[...,None].clamp_min(1e-12))
    residual=qbar*cov.float()[:,None]
    own=(torch.arange(t,device=q.device)+p-t)[None].expand(h,-1)
    z=mass.detach().clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=(weighted[...,None]*conditional).sum(2)/den;allvalue=(mass[...,None]*conditional).sum(2)
    missing=allvalue-selected;diff=conditional-selected[:,:,None];energy=selected.square().sum(-1,keepdim=True).clamp_min(.01)
    align=(diff*missing[:,:,None]).sum(-1)/energy;mag=diff.square().sum(-1)/energy
    extra=torch.stack((mask.float(),(mass/den).clamp(0,2),align.clamp(-10,10)/5,mag.clamp(0,10)/5,(mass*align).clamp(-2,2),(mass*mag).clamp(0,2)),-1)
    s=score(mass,torch.cat((feat,extra),-1),m.route)
    contribution=mass[...,None]*residual;error=(contribution*(~mask)[...,None]).sum(2)
    # Add reduces predicted error by 2<E,e>-||e||²; dropping costs 2<E,e>+||e||².
    residual_energy=error.square().sum(-1,keepdim=True).clamp_min(1e-8)
    gain=(2*(error[:,:,None]*contribution).sum(-1)+(2*mask.float()-1)*contribution.square().sum(-1))/residual_energy
    s=s+m.residual_gate.tanh()[:,None,None]*gain.clamp(-5,5)
    with torch.no_grad():
        new,_=correct(mask,s,own)
        def lut(x):return x.int().topk(keep,-1).indices.sort(-1).values[None].int()
        chosen=lut(new)
        if not training:return chosen,None,None,None
        rem=s.masked_fill(~mask,float('inf')).clone();rem.scatter_(-1,own[...,None],float('inf'));drop=rem.argmin(-1);add=s.masked_fill(mask,-float('inf')).argmax(-1)
        alt=mask.clone().scatter_(-1,drop[...,None],False).scatter_(-1,add[...,None],True);baseids=lut(mask);altids=lut(alt)
    margin=s.gather(-1,add[...,None]).squeeze(-1)-s.gather(-1,drop[...,None]).squeeze(-1)
    return chosen,baseids,altids,margin
