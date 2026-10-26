"""No full content tensor: tiled moments or Gram aggregate statistics."""
import torch
import triton as tr
from ..fast_route import FastRoute
from ..opt2.score import _content
from ..opt2.streaming import _reduce_moments
from ..opt2.candidates import candidate_sides_fast
from .kernels import mm,_weights_energy,_gram_stats,_candidate_content
from .tiled_kernel import _tile_moments
from .actions import _candidate_actions
from .packed import prepare_packed,_packed_score,_packed_weights,_packed_candidates
from .streamgram import prepare_stream,first as stream_first,means as stream_means


class CompactRoute:
    def __init__(self,method='tiled',precision='tf32x3',score_precision='ieee',gp=8,tile=32,batch=512):
        self.method=method;self.precision=precision;self.score_precision=score_precision
        self.gp=gp;self.tile=tile;self.batch=batch
        base=FastRoute(True,True,batch=batch)
        self.mask=base.mask
        self.candidates=torch.compile(candidate_sides_fast,fullgraph=True,dynamic=False)
        self.prepare_packed=torch.compile(prepare_packed,fullgraph=True,dynamic=False)
        self.prepare_stream=torch.compile(prepare_stream,fullgraph=True,dynamic=False)

    def begin(self,prepared,sketch):
        state=dict(prepared=prepared)
        if prepared[1][0][2] is None:return state
        if self.method=='streamgram':
            state.update(self.prepare_stream(prepared));state['costs']=sketch.costs
        elif self.method=='packed':
            state.update(self.prepare_packed(prepared))
        elif self.method=='gram':
            state['grams']=[v@v.transpose(-1,-2) for _,_,v in prepared[1]]
            state['value']=torch.cat([v.flatten(1,2) for _,_,v in prepared[1]],1)
        return state

    def score(self,q,state):
        basis,projected=state['prepared']
        h,n,_=q.shape;b=sum(lc.shape[1] for _,lc,_ in projected)
        qq=torch.bmm(q.float(),basis)
        score=torch.empty((h,n,b),device=q.device,dtype=torch.float32)
        if self.method in ['packed','streamgram']:
            key=state['key'];r=key.shape[1]
            raw=torch.bmm(qq,key.transpose(1,2)) if self.score_precision=='ieee' else mm(qq,key.transpose(1,2),self.score_precision)
            if self.method=='streamgram':
                score,weights,energy=stream_first(raw,state,state['costs'],b,self.precision,self.tile)
                return score,(raw,weights,energy)
            _packed_score[(tr.cdiv(n,32),b,h)](raw,state['lc'],state['starts'],state['counts'],score,n,b,r,32,
                num_warps=4,enable_fp_fusion=True)
            return score,raw
        raw=[];offset=0
        for key,lc,_ in projected:
            nb,p=lc.shape[1:]
            logits=torch.bmm(qq,key.transpose(1,2)) if self.score_precision=='ieee' else mm(qq,key.transpose(1,2),self.score_precision)
            raw.append(logits)
            _content[(tr.cdiv(n,32),nb,h)](logits,lc,None,score,None,n,nb,p,b,offset,32,max(16,p),False,
                num_warps=4,enable_fp_fusion=True)
            offset+=nb
        return score,raw

    def stats(self,q,sketch,score,mask,rem,add,raw,state):
        h,n,b=score.shape;projected=state['prepared'][1]
        stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
        cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
        if self.method=='tiled':
            gp=self.gp;groups=sum(tr.cdiv(lc.shape[1],gp) for _,lc,_ in projected)
            partial=torch.empty((h*n,groups,68),device=q.device,dtype=torch.float32)
            offset=goff=0
            for logits,(_,lc,value) in zip(raw,projected):
                nb,p=lc.shape[1:]
                _tile_moments[(tr.cdiv(n,self.tile),tr.cdiv(nb,gp),h)](logits,lc,value,mask,sketch.costs,rem,add,
                    partial,cv,n,nb,p,b,offset,groups,goff,gp,self.tile,max(16,p),False,self.precision,
                    num_warps=4,num_stages=1,enable_fp_fusion=True)
                offset+=nb;goff+=tr.cdiv(nb,gp)
            _reduce_moments[(h*n,)](partial,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),
                groups,tr.next_power_of_2(groups),num_warps=4,enable_fp_fusion=False)
        elif self.method=='streamgram':
            logits,weights,energy=raw;r=logits.shape[-1]
            means=stream_means(weights,mask,state,self.precision)
            _packed_candidates[(h*n,)](logits,state['lc'],state['value'],state['starts'],state['counts'],rem,add,cv,n,r,
                num_warps=4,enable_fp_fusion=True)
            _gram_stats[(h*n,)](means,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),
                num_warps=4,enable_fp_fusion=False)
        elif self.method=='packed':
            r=state['value'].shape[1]
            weights=torch.empty((h,2*n,r),device=q.device,dtype=torch.float16 if self.precision=='fp16'else torch.float32)
            energy=torch.empty_like(score)
            _packed_weights[(tr.cdiv(n,32),b,h)](raw,state['lc'],state['grams'],state['starts'],state['counts'],
                mask,sketch.costs,weights,energy,n,b,r,32,self.precision,num_warps=4,enable_fp_fusion=True)
            _packed_candidates[(h*n,)](raw,state['lc'],state['value'],state['starts'],state['counts'],rem,add,cv,n,r,
                num_warps=4,enable_fp_fusion=True)
            means=mm(weights,state['value'],self.precision)
            _gram_stats[(h*n,)](means,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),
                num_warps=4,enable_fp_fusion=False)
        elif self.method=='gram':
            r=state['value'].shape[1]
            weights=torch.empty((h,2*n,r),device=q.device,dtype=torch.float16 if self.precision=='fp16'else torch.float32)
            energy=torch.empty_like(score)
            offset=roff=0
            for logits,(_,lc,value),gram in zip(raw,projected,state['grams']):
                nb,p=lc.shape[1:]
                _weights_energy[(tr.cdiv(n,32),nb,h)](logits,lc,gram,mask,sketch.costs,weights,energy,
                    n,nb,p,b,offset,r,roff,32,max(16,p),self.precision,num_warps=4,enable_fp_fusion=True)
                _candidate_content[(h*n,)](logits,lc,value,rem,add,cv,n,nb,p,offset,max(16,p),num_warps=4,enable_fp_fusion=True)
                offset+=nb;roff+=nb*p
            means=mm(weights,state['value'],self.precision)
            _gram_stats[(h*n,)](means,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),
                num_warps=4,enable_fp_fusion=False)
        else:raise ValueError(self.method)
        return stats,cv

    def actions(self,q,sketch,score,mask,rem,add,beta,step,stats,cv,head):
        h,n,b=score.shape;out=torch.empty_like(mask)
        _candidate_actions[(h*n,)](stats,score,cv,mask,sketch.costs,sketch.origins,rem,add,
            head.w1,head.b1,head.w2,head.b2,head.mean,head.std,step,out,n,b,tr.next_power_of_2(b),
            sketch.history_tokens,sketch.current_tokens,beta,self.precision,num_warps=4,enable_fp_fusion=False)
        return out

    def run(self,layer,q,sketch,state,step,router):
        if router.kind=='U-Sol':
            from ..opt2_precision import fma_score
            score,_=fma_score(q,*state['prepared'])
            return self.mask(score,router.beta)
        score,raw=self.score(q,state)
        mask=self.mask(score,router.beta)
        if router.kind=='U-Sol':return mask
        rem,add=self.candidates(score,mask,sketch.origins,router.beta,128,1,q.shape[1])
        stats,cv=self.stats(q,sketch,score,mask,rem,add,raw,state)
        return self.actions(q,sketch,score,mask,rem,add,router.beta,step,stats,cv,router.heads[f'{layer}_{router.kind}'])

    def __call__(self,layer,q,sketch,prepared,step,router):
        if not isinstance(step,torch.Tensor):step=q.new_tensor(float(step),dtype=torch.float32)
        return self.run(layer,q,sketch,self.begin(prepared,sketch),step,router)
