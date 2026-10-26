"""Verify absent candidates, singleton/all-selected sets and action ties."""
import argparse,copy
from pathlib import Path
import torch
from .common import atomic_json
from .learning import ActionHead
from .opt5.network import OptimizedRoute,OPTIONS as BASE_OPTIONS
from .opt6_guarded import GuardedHeadRoute
from .opt6.debug import scores as head_scores

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:17];sketch=data['sketch'];beta=data['beta'];step=q.new_tensor(3.,dtype=torch.float32)
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    ref=OptimizedRoute(**BASE_OPTIONS);new=GuardedHeadRoute();state=ref.begin(data['prepared'],sketch)
    score,raw=ref.score(q,state);base=ref.mask(score,beta);rows=[]
    for case in ['no_candidates','remove_only','add_only','singleton','all_selected','zero_head','equal_positive']:
        mask=base.clone();hh=copy.deepcopy(head)
        if case=='singleton':mask.zero_();mask.scatter_(-1,score.argmax(-1,keepdim=True),True)
        if case=='all_selected':mask.fill_(True)
        rem,add=ref.candidates(score,mask,sketch.origins,beta,128,1,q.shape[1])
        if case in ['no_candidates','add_only']:rem.fill_(-1)
        if case in ['no_candidates','remove_only']:add.fill_(-1)
        if case in ['zero_head','equal_positive']:
            for param in hh.parameters():param.zero_()
            if case=='equal_positive':hh.b2.fill_(1.)
        st,cv=ref.stats(q,sketch,score,mask,rem,add,raw,state)
        args=(q,sketch,score,mask,rem,add,beta,step,st,cv,hh)
        expected=ref.actions(*args);actual=new.actions(*args)
        pred,dm=head_scores(new,*args);rp,rd=head_scores(new,*args,reference=True)
        assert torch.equal(dm,actual)and torch.equal(rd,expected)and torch.equal(actual,expected)and actual.any(-1).all(),case
        assert torch.isfinite(pred).all(),case
        if case in ['no_candidates','zero_head']:assert torch.equal(actual,mask),case
        row=dict(case=case,head_queries=q.shape[0]*q.shape[1],mask_equal=True,nonempty=True,prediction_max_abs=float((pred-rp).abs().max()))
        rows.append(row);print(row,flush=True)
    atomic_json(root/'optimization6/guard_boundary_checks.json',dict(status='passed',cases=rows))

if __name__=='__main__':main()
