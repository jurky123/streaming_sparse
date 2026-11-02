import math,torch
from .gap_pool_reduce import weighted
from .gap_direct_center_kernel import direct_centers as centers
from .high_sparse_learn import features,correct
from .high_sparse_gain import score
from .gap_pool_weights_kernel import weights
from .gap_direct_summary_ops import direct_pool as reference_pool
def raw_pool(self,k,v,cheap=False):
    if not cheap:return reference_pool(self,k,v,cheap=False)
    h,p,_,d=k.shape
    km=k.mean(2,keepdim=True,dtype=torch.float32);vm=v.mean(2,keepdim=True,dtype=torch.float32);kc=k-km
    if cheap:
        f=torch.einsum('hptd,hrd->hptr',kc,self.bank)/128**.5
        f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    else:f=self.group_features(k.float(),v.float())
    f=f.reshape(h,p,4,16,32);kg=k.reshape(h,p,4,16,d);vg=v.reshape_as(kg)
    a=weights(k,v,f,self.pool_k,self.pool_v,self.pool_feature)
    pk,pv=weighted(k,v,a,block=32)
    # Preserve page-level K mean and encoded V mean. No new cache bytes.
    pk=pk-pk.mean(2,keepdim=True)+km
    basis=self.basis.to(torch.bfloat16).float()
    codes=torch.einsum('hprd,hdk->hprk',pv-self.center[:,None,None],basis)
    base=torch.einsum('hpd,hdk->hpk',vm.squeeze(2)-self.center[:,None],basis).to(torch.bfloat16)
    haar=self._cached_haar
    detail=(torch.einsum('kr,hprd->hpkd',haar,codes)/4).to(torch.bfloat16)
    return pk.contiguous(),base.contiguous(),detail.contiguous()
