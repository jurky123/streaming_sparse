"""Sample exact softmax mass captured by the actual per-tile LUT on a dense teacher."""
import math
from pathlib import Path
import torch
import numpy as np

class PageDiagnostic:
    def __init__(self,rt):
        self.rt=rt
        self.op=rt.packed_attention
        self.values=torch.zeros(7,5,device='cuda',dtype=torch.float64)

    def __getattr__(self,name):return getattr(self.op,name)

    def __call__(self,store,q,k,v,s,basis,ids,workspace,**kwargs):
        y=self.op(store,q,k,v,s,basis,ids,workspace,**kwargs)
        c,l,st=[self.rt.engine.current[n] for n in ('chunk','layer','step')]
        h=torch.arange(12,device=q.device)
        row=(h*113+c*67+st*43+l*79)%4608
        query=q[0,row,h].float()[:,None]
        logits=torch.bmm(query,k[0].permute(1,2,0).float())/math.sqrt(128)
        probs=logits.softmax(-1)
        selected=ids[h,row//64].long()
        mask=torch.zeros((12,k.size(1)//64),device=q.device,dtype=torch.bool).scatter_(1,selected,True)
        kept=probs.reshape(12,-1,64)*mask[:,:,None]
        mass=kept.sum((1,2))
        vals=v[0].transpose(0,1).float()
        target=torch.bmm(probs,vals)[:,0]
        exact=torch.bmm(kept.reshape(12,1,-1),vals)[:,0]/mass[:,None]
        mixed=y[0,row,h].float()
        self.values[c]+=torch.stack((mass.sum(),mass.new_tensor(12),
                        (exact-target).square().sum(),(mixed-target).square().sum(),target.square().sum()))
        return y
