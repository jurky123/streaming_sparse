import torch
from .improve_runtime import FastRuntime
from .response_hard import features,balanced
from .cache_grouping import reorder
from .rollout_kernels import aggregate
from .improve_scatter_kernel import copy
class ScatterRuntime(FastRuntime):
    def __init__(self,engine):super().__init__(engine,'scatter_fast')
    def commit(self,l,k,v,cache):
        kh=k[0].transpose(0,1).float();vh=v[0].transpose(0,1)
        x=features(kh,vh,self.bank[l],'response');order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
        inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
        order=order.reshape(12,72,64).gather(2,inner)[:,self.page_order].flatten(1).contiguous()
        kr=reorder(kh,order);proto=aggregate(kr,self.weights.get(l))
        copy(k,v,order,cache,self.write[l],self.cap)
        slots=(torch.arange(72,device=k.device)+self.write[l])%self.cap;self.pk[l].index_copy_(1,slots,proto)
        self.write[l]=(self.write[l]+72)%self.cap;self.n[l]=min(self.cap,self.n[l]+72);self.commits[l]+=1
