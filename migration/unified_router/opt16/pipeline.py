"""Complete opt-in grouped attention block; no mutation of Opt14/model defaults."""
from dataclasses import dataclass
import torch
from ..core import build
from ..fast_route import prepare as project_prototypes
from ..opt9.segmented_attention import attention
from .grouping import order,gather,restore
from .route import prepare,route

def _build(k,v,nh,parameters):
    sketch=build(k[:,:nh],v[:,:nh],k[:,nh:],v[:,nh:],parameters,need_value=False)
    return prepare(project_prototypes(sketch,'U-Sol'))

@dataclass
class PackedKV:
    k: torch.Tensor
    v: torch.Tensor
    order: torch.Tensor
    state: dict
    history_tokens: int

class GroupedRouter:
    def __init__(self,parameters,qtree,ktree,keep=.15,candidate_fraction=.5,compile=True):
        self.parameters=parameters;self.qtree=qtree;self.ktree=ktree
        self.keep=keep;self.candidate_fraction=candidate_fraction
        self.build=torch.compile(_build,fullgraph=True) if compile else _build

    def pack_kv(self,k,v,history_tokens,group=True):
        if not (0 < history_tokens < k.shape[1] and history_tokens%4680==0 and k.shape[1]-history_tokens<=4680):
            raise ValueError('Grouped sparse block requires complete 4680-token history chunks and one current chunk; first chunk is Dense')
        if k.shape!=v.shape or k.shape[-1]!=128:
            raise ValueError('K/V must have matching [heads,tokens,128] shapes')
        # Chunk-local permutation preserves age/P and cache ownership boundaries.
        indices=[]
        for start in range(0,k.shape[1],4680):
            end=min(start+4680,k.shape[1])
            ix=order(k[:,start:end],self.ktree) if group else torch.arange(end-start,device=k.device,dtype=torch.int32)[None].expand(k.shape[0],-1)
            indices.append(ix+start)
        ix=torch.cat(indices,1);kk=gather(k,ix);vv=gather(v,ix)
        return PackedKV(kk,vv,ix,self.build(kk,vv,history_tokens,self.parameters),history_tokens)

    def run(self,q,kv,group=True,return_aux=False):
        ix=order(q,self.qtree) if group else torch.arange(q.shape[1],device=q.device,dtype=torch.int32)[None].expand(q.shape[0],-1)
        qq=gather(q,ix);mask,aux=route(qq,kv.state,self.keep,self.candidate_fraction,True)
        nh=kv.history_tokens
        y=attention(qq,kv.k[:,:nh],kv.v[:,:nh],kv.k[:,nh:],kv.v[:,nh:],mask,pack_v=True,fuse_pack=True)
        out=restore(y,ix)
        return (out,dict(**aux,mask=mask,q_order=ix,packed_q=qq)) if return_aux else out

    def __call__(self,q,k,v,history_tokens,group_q=True,group_k=True,return_aux=False):
        kv=self.pack_kv(k,v,history_tokens,group_k)
        result=self.run(q,kv,group_q,return_aux)
        return (result,kv) if return_aux else result
