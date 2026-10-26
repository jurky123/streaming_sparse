"""Train the actual shared32 Opt19 kernel, with a tile-aware biased STE.

The loss forward is the real INT8-QK/BF16-PV output. All32 members contribute
to a tile's differentiable probability aggregation. Dense page statistics are
used only for the backward Jacobian, never as online selection features.
"""
import math
import torch
import torch.nn.functional as F
from ..opt17.model import JointGroupedRouter
from ..opt16.pipeline import GroupedRouter
from ..opt16.grouping import gather
from ..opt19.pipeline import Pipeline
from ..opt12.model import ActualAttention,page_statistics
from ..opt12.precision import align
from ..core import build


class TileRouter(JointGroupedRouter):
    def __init__(self,m,layer,variant):
        super().__init__(m,layer,variant)
        self.budget=self.options['tile_budget'];self.group_size=32

    @torch.no_grad()
    def deployed(self,q,raw,targets,fraction=None,representatives=None,**unused):
        if fraction is not None and fraction!=self.options['candidate_fraction']:raise ValueError('Opt20 candidate budget is fixed')
        if representatives is not None and representatives!=self.options['representatives']:raise ValueError('Opt20 representative count is fixed')
        # Training changes parameters after every step. No derived KV/prototype
        # cache is allowed to survive across optimizer steps or raw events.
        self.pipeline=GroupedRouter(self.dictionary(),self.tree('q'),self.tree('k'),compile=False)
        runner=Pipeline(self,raw,32,self.budget,'shared','fp16')
        full,d=runner.run(q,raw['kc'],raw['vc'],True)
        qi=d['q_order'];h,n,_=q.shape;kv=d['kv'];nk=kv.k.shape[1];b=kv.state['key'].shape[1]
        inv=torch.empty_like(qi);inv.scatter_(1,qi.long(),torch.arange(n,device=q.device,dtype=qi.dtype)[None].expand_as(qi))
        pos=inv[:,targets].long();group=pos//32;head=torch.arange(h,device=q.device)[:,None]
        selected=d['selection'][head,group].long();candidates=d['candidates'][head,group].long()
        mask=torch.zeros((h,targets.numel(),b),device=q.device,dtype=torch.float32).scatter_(-1,selected,1.)
        costs=torch.where(d['selection']%74==73,8.,64.).sum(-1)
        qcost=torch.full((costs.shape[1],),32.,device=q.device);qcost[-1]=n-(len(qcost)-1)*32
        keep=(costs*qcost).sum()/(h*n*nk)
        executed=q.new_tensor(d['selection'].numel()*32*64/(h*n*nk),dtype=torch.float32)
        d.update(packed_q=gather(q,qi),position=pos,group=group,selected=selected,
                 target_candidates=candidates,mask=mask,keep=keep,executed=executed,keep_pages=selected.shape[-1])
        return full[:,targets].float(),d

    def forward(self,q,raw,targets,temperature=.5):
        with torch.no_grad():
            actual,d=self.deployed(q,raw,targets)
            kv=d['kv'];nh=kv.history_tokens
            grouped=dict(kh=kv.k[:,:nh],vh=kv.v[:,:nh],kc=kv.k[:,nh:],vc=kv.v[:,nh:])
            logz,value=page_statistics(q[:,targets],grouped)
            h,n,_=q.shape;t=targets.numel();b=d['mask'].shape[-1]
            positions=d['group'][...,None]*32+torch.arange(32,device=q.device)[None,None]
            valid=positions<n;safe=positions.clamp_max(n-1)
            member_q=gather(d['packed_q'],safe.flatten(1)).float()
            actual_fine=gather(d['fine'],safe.flatten(1)).reshape(h,t,32,-1)
            hard_prob=actual_fine.softmax(-1)*valid[...,None]
            hard_importance=torch.zeros((h,t,b),device=q.device).scatter_(-1,d['target_candidates'],hard_prob.sum(2))
        sketch=build(grouped['kh'],grouped['vh'],grouped['kc'],grouped['vc'],self.dictionary(),need_value=False)
        score=self.differentiable_scores(member_q,sketch).reshape(h,t,32,b)
        cid=d['target_candidates'][:,:,None].expand(-1,-1,32,-1)
        exact_score=score.detach().clone().scatter_(-1,cid,actual_fine)
        score=align(score,exact_score)
        r=self.options['representatives'];representatives=torch.arange(r,device=q.device)*(32//r)+(32//r)//2-1
        coarse=score[:,:,representatives].amax(2)
        chosen_coarse=coarse.gather(-1,d['target_candidates'])
        boundary_page=d['target_candidates'].gather(-1,chosen_coarse.detach().argmin(-1,keepdim=True))
        # Backward-only smooth normalization in log space. A hard zero gate
        # after exp could leave an almost-zero denominator when an excluded
        # page dominates, amplifying gradients by up to 1e20. The hard tile
        # importance and real attention forward are still restored below.
        log_candidates=F.logsigmoid((coarse-coarse.gather(-1,boundary_page))/temperature)
        prob=(score+log_candidates[:,:,None]).softmax(-1)
        importance=(prob*valid[...,None]).sum(2)
        importance=align(importance,hard_importance)
        boundary_index=hard_importance.gather(-1,d['selected']).argmin(-1,keepdim=True)
        boundary=d['selected'].gather(-1,boundary_index)
        log_importance=(importance+1e-6).log()
        soft=((log_importance-log_importance.gather(-1,boundary))/temperature).sigmoid()
        gate=align(soft,d['mask'])
        output=ActualAttention.apply(gate,actual,logz,value)
        return output,dict(keep=d['keep'],executed=d['executed'],gate_mean=soft.mean())
