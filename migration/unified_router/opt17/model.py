"""Hard Opt17 forward; FP32 backward surrogate; alternating discrete trees.

The loss value is the actual deployed output error with full Q grouping context.
Gradients through discrete coarse/fine gates are biased STE, not exact gradients.
Tree parameters are buffers updated separately by Train-only proposal evaluation.
"""
from pathlib import Path
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..core import build
from ..opt16.grouping import Tree,fit_tree,order,gather,restore
from ..opt16.pipeline import GroupedRouter
from ..opt9.segmented_attention import attention
from ..opt12.precision import align,HalfBmmFP32Backward,fp16_round
from ..opt12.model import ActualAttention,page_statistics
from .route import route

class JointGroupedRouter(nn.Module):
    def __init__(self,m,layer,variant):
        super().__init__();self.layer=layer;self.variant=variant;self.options=m['variants'][variant]
        self.keep=m['target_keep'];self.depth=m['tree_depth']
        p=torch.load(Path(m['parent_experiment'])/f'action_basis/layer{layer:02d}.pt',map_location='cpu',weights_only=True)['parameters']
        self.routing=nn.ParameterDict({k:nn.Parameter(v.clone())for k,v in p.items()if k not in ('key_basis','value_basis')})
        self.routing['key_projection']=nn.Parameter(p['key_basis'][...,:64].clone())
        self.register_buffer('key_tail',p['key_basis'][...,64:].clone());self.register_buffer('value_basis',p['value_basis'].clone())
        for side in ['q','k']:
            self.register_buffer(side+'_tree_weight',torch.zeros(12,2**self.depth-1,128))
            self.register_buffer(side+'_tree_threshold',torch.zeros(12,2**self.depth-1))
        self.pipeline=None

    def dictionary(self):
        p={k:v for k,v in self.routing.items()if k!='key_projection'}
        return dict(p,key_basis=torch.cat((self.routing['key_projection'],self.key_tail),-1),value_basis=self.value_basis)

    def tree(self,side):return Tree(getattr(self,side+'_tree_weight'),getattr(self,side+'_tree_threshold'),self.depth)

    @torch.no_grad()
    def set_trees(self,qt,kt):
        for side,tree in [('q',qt),('k',kt)]:
            getattr(self,side+'_tree_weight').copy_(tree.weight);getattr(self,side+'_tree_threshold').copy_(tree.threshold)

    @torch.no_grad()
    def propose_trees(self,q,k,projected=False):
        if projected:
            basis=self.routing['key_projection'];qopp=torch.bmm(torch.bmm(q.float(),basis),basis.transpose(1,2));kopp=torch.bmm(torch.bmm(k.float(),basis),basis.transpose(1,2))
        else:qopp=q;kopp=k
        return fit_tree(q,kopp,self.depth),fit_tree(k,qopp,self.depth)

    @torch.no_grad()
    def deployed(self,q,raw,targets,fraction=None,representatives=None):
        if self.pipeline is None:self.pipeline=GroupedRouter(self.dictionary(),self.tree('q'),self.tree('k'),compile=True)
        self.pipeline.parameters=self.dictionary()
        k=torch.cat((raw['kh'],raw['kc']),1);v=torch.cat((raw['vh'],raw['vc']),1);nh=raw['kh'].shape[1]
        kv=self.pipeline.pack_kv(k,v,nh,True)
        qi=order(q,self.tree('q'));qq=gather(q,qi)
        frac=self.options['candidate_fraction']if fraction is None else fraction
        reps=self.options['representatives']if representatives is None else representatives
        mask,aux=route(qq,kv.state,self.keep,frac,True,reps)
        y=attention(qq,kv.k[:,:nh],kv.v[:,:nh],kv.k[:,nh:],kv.v[:,nh:],mask,pack_v=True,fuse_pack=True)
        output=restore(y,qi).float()[:,targets]
        inv=torch.empty_like(qi);inv.scatter_(1,qi.long(),torch.arange(q.shape[1],device=q.device,dtype=qi.dtype)[None].expand_as(qi))
        pos=inv[:,targets].long();head=torch.arange(q.shape[0],device=q.device)[:,None]
        selected=mask[head,pos];group=pos//32
        ids=aux['candidates'][head,group].long();fine=aux['scores'][head,pos]
        coarse=aux['coarse'][head,group] if aux['coarse'] is not None else None
        costs=torch.full((mask.shape[-1],),64.,device=q.device);costs[73::74]=8.
        padded=F.pad(mask,(0,0,0,(-q.shape[1])%32))
        executed=padded.reshape(q.shape[0],-1,32,mask.shape[-1]).any(2).sum()*32*64/(q.shape[0]*q.shape[1]*k.shape[1])
        return output,dict(mask=selected,full_mask=mask,candidates=ids,fine=fine,coarse=coarse,group=group,
            q_order=qi,packed_q=qq,kv=kv,position=pos,costs=costs,keep=(mask*costs).sum()/(q.shape[0]*q.shape[1]*k.shape[1]),executed=executed,
            keep_pages=aux['keep_pages'],fraction=frac,representatives=reps)

    def differentiable_scores(self,q,sketch):
        basis=sketch.key_basis[...,:64];qq=HalfBmmFP32Backward.apply(q,basis);out=[]
        for p in sketch.parts:
            h,b,np,_=p.key.shape
            keys=torch.bmm(p.key.flatten(1,2),basis)
            logits=HalfBmmFP32Backward.apply(qq,keys.transpose(1,2)).reshape(h,q.shape[1],b,np)/math.sqrt(128)
            lc=p.count.clamp_min(1e-30).log().masked_fill(p.count<=1e-8,-torch.inf)
            out.append((logits+lc[:,None]).logsumexp(-1))
        return torch.cat(out,2)

    def forward(self,q,raw,targets,temperature=.5):
        with torch.no_grad():
            actual,d=self.deployed(q,raw,targets)
            kv=d['kv'];nh=kv.history_tokens
            grouped=dict(kh=kv.k[:,:nh],vh=kv.v[:,:nh],kc=kv.k[:,nh:],vc=kv.v[:,nh:])
            logz,value=page_statistics(q[:,targets],grouped)
        sketch=build(grouped['kh'],grouped['vh'],grouped['kc'],grouped['vc'],self.dictionary(),need_value=False)
        h,t=targets.numel(),d['representatives']
        positions=(d['group'][...,None]*32+torch.arange(t,device=q.device)[None,None]*(32//t)+(32//t)//2-1).clamp_max(q.shape[1]-1)
        lm=gather(d['packed_q'],positions.flatten(1)).float()
        allq=torch.cat((q[:,targets].float(),lm),1)
        scores=self.differentiable_scores(allq,sketch)
        fine=scores[:,:h];coarse=scores[:,h:].reshape(q.shape[0],h,t,-1).amax(2)
        actual_fine=fine.detach().clone().scatter(-1,d['candidates'],d['fine'])
        fine=align(fine,actual_fine);coarse=align(coarse,d['coarse'])
        # Detach the rank, differentiate the boundary score: invariant to row shifts.
        fine_boundary=d['fine'].argsort(-1,descending=True)[...,d['keep_pages']-1:d['keep_pages']]
        fine_page=d['candidates'].gather(-1,fine_boundary)
        fc=fine.gather(-1,fine_page)
        coarse_page=d['candidates'].gather(-1,d['coarse'].gather(-1,d['candidates']).argmin(-1,keepdim=True))
        cc=coarse.gather(-1,coarse_page)
        soft=(fine.sub(fc).div(temperature).sigmoid()*coarse.sub(cc).div(temperature).sigmoid())
        gate=align(soft,d['mask'].float())
        out=ActualAttention.apply(gate,actual,logz,value)
        return out,dict(keep=d['keep'],executed=d['executed'],gate_mean=soft.mean())
