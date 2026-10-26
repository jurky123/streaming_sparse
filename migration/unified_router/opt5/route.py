"""Two-pass route; independently normalizes and selects every query."""
import torch
import triton as tr
from ..opt4.route import CompactRoute
from .kernels import prepare_flash,_score_energy,_recompute_means,_recompute_candidates
from .reduce import _partial_stats
from .candidate_simt import candidate_simt


class FlashRoute(CompactRoute):
    def __init__(self,qm=64,gp=4,br=64,splits=4,cs=16,energy='gram',candidate='mma',batch=512):
        super().__init__('streamgram','fp16','fp16',batch=batch)
        self.candidate=candidate;self.qm=qm;self.page_group=gp;self.br=br;self.splits=splits;self.cs=cs;self.energy=energy
        self.prepare_flash=torch.compile(prepare_flash,fullgraph=True,dynamic=False)

    def begin(self,prepared,sketch):
        state=dict(prepared=prepared,costs=sketch.costs)
        if prepared[1][0][2]is not None:state.update(self.prepare_flash(prepared))
        return state

    def score(self,q,state):
        h,n,_=q.shape;b=state['costs'].numel()
        qq=torch.bmm(q.float(),state['prepared'][0]).half()
        score=torch.empty((h,n,b),device=q.device,dtype=torch.float32)
        energy=torch.empty_like(score)
        _score_energy[(tr.cdiv(n,self.qm),tr.cdiv(b,self.page_group),h)](qq,state['padded_key'],state['padded_lc'],state['grams'],state['padded_value'],
            score,energy,n,b,self.qm,self.page_group,self.energy=='content',num_warps=4,num_stages=1,enable_fp_fusion=True)
        return score,(qq,energy)

    def stats(self,q,sketch,score,mask,rem,add,raw,state):
        qq,energy=raw;h,n,b=score.shape;r=state['key16'].shape[1]
        part=torch.empty((h*n,self.splits,64),device=q.device,dtype=torch.float32)
        stats=torch.empty((h*n,80),device=q.device,dtype=torch.float32)
        cv=torch.empty((h*n,16,32),device=q.device,dtype=torch.float32)
        chunk=tr.cdiv(tr.cdiv(r,self.br),self.splits)*self.br
        _recompute_means[(tr.cdiv(n,self.qm),self.splits,h)](qq,state['key16'],state['value16'],state['lc'],state['page_index'],sketch.costs,
            score,mask,part,n,b,r,self.qm,self.br,self.splits,chunk,num_warps=4,num_stages=2,enable_fp_fusion=True)
        (candidate_simt if self.candidate=='simt' else _recompute_candidates)[(h*n,16//self.cs)](qq,state['key16'],state['value'],state['lc'],state['starts'],state['counts'],rem,add,cv,
            n,r,self.cs,num_warps=4,num_stages=1,enable_fp_fusion=True)
        _partial_stats[(h*n,)](part,energy,score,mask,sketch.costs,sketch.origins,stats,n,b,tr.next_power_of_2(b),
            self.splits,tr.next_power_of_2(self.splits),num_warps=4,enable_fp_fusion=False)
        return stats,cv
