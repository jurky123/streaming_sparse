import math,torch
from .gap_pool_reduce import weighted
from .gap_group_center_kernel import group_centers as centers
from .high_sparse_learn import features,correct
from .high_sparse_gain import score
def layout_pool(self,k,v,cheap=False):
    h,p,_,d=k.shape;k=k.float();v=v.float()
    km,vm,kc,kgc,vgc=centers(k,v)
    if cheap:
        f=torch.einsum('hptd,hrd->hptr',kc,self.bank)/128**.5
        f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    else:f=self.group_features(k,v)
    f=f.reshape(h,p,4,16,32);kg=k.reshape(h,p,4,16,d);vg=v.reshape_as(kg)
    z=torch.einsum('hprtd,hrd->hprt',kgc,self.pool_k)
    z=z+torch.einsum('hprtd,hrd->hprt',vgc,self.pool_v)+torch.einsum('hprtf,hrf->hprt',f,self.pool_feature)
    z=z-z.mean(-1,keepdim=True);a=(2*torch.tanh(z/2)).softmax(-1)
    pk,pv=weighted(k,v,a,block=32)
    # Preserve page-level K mean and encoded V mean. No new cache bytes.
    pk=pk-pk.mean(2,keepdim=True)+km
    basis=self.basis.to(torch.bfloat16).float()
    codes=torch.einsum('hprd,hdk->hprk',pv-self.center[:,None,None],basis)
    base=torch.einsum('hpd,hdk->hpk',vm.squeeze(2)-self.center[:,None],basis).to(torch.bfloat16)
    haar=self._cached_haar
    detail=(torch.einsum('kr,hprd->hpkd',haar,codes)/4).to(torch.bfloat16)
    return pk.contiguous(),base.contiguous(),detail.contiguous()
