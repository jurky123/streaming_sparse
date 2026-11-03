"""Paired same-QKV/same-page ablations on accepted trajectories; no QKV dumps."""
import argparse,json,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_gain_vector64_runtime import GapGainVector64Runtime
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .review_v1.model import Model
from .review_v1.ops import Summary,SummaryOps,AttentionOps,RouteOps,Config
from .review_v1.ablation import change_summary,mass_scores
ROOT=Path('artifacts/review_v1_20261102/ablation/local')
BASE='artifacts/four_gaps_20261031/shared/epoch1.pt'
FINAL='artifacts/review_v1_20261102/route_expanded/model.pt'
PRE='artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();root=ROOT/a.case;lock=RunLock(root);rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        identity=dict(case=a.case,seed=101,chunks=[2,5,8],steps=[0,2],layers=[1,8,15,22,28,29],tiles=[0,36],probes=16,source=digest(__file__),ops=digest('local_experiments/review_v1/ops.py'),ablation=digest('local_experiments/review_v1/ablation.py'),models={p:digest(p) for p in (BASE,FINAL,PRE)},scope='paired same QKV/physical pages; no regrouping inference from this local test')
        if (root/'identity.json').exists():assert json.loads((root/'identity.json').read_text())==identity
        else:atomic(root/'identity.json',identity)
        if (root/'complete.json').exists():return
        models={name:Model(BASE) for name in ('initial','pretrained','final')}
        models['pretrained'].load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model'])
        models['final'].load_state_dict(torch.load(FINAL,map_location='cuda',weights_only=False)['model'])
        engine=Engine();rt=GapGainVector64Runtime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=False)['model']);original=rt.decision
        tiles=torch.tensor([0,36],device='cuda');offsets=torch.linspace(0,63,16,device='cuda').round().long()
        def decision(l,q,pk,base,detail,cov,vvar):
            result=original(l,q,pk,base,detail,cov,vvar);chunk=engine.current['chunk'];step=engine.current['step'];dest=root/f'c{chunk}_l{l}_s{step}.json'
            if chunk not in [2,5,8] or step not in [0,2] or l not in [1,8,15,22,28,29] or dest.exists():return result
            disk_guard(root,50,1);atomic(root/'status.json',dict(status='evaluating',chunk=chunk,step=step,layer=l))
            p=pk.shape[1];h=12;k=engine.cache[l]['k'][0,:p*64].transpose(0,1).reshape(h,p,64,128);v=engine.cache[l]['v'][0,:p*64].transpose(0,1).reshape_as(k)
            qs=q[0].transpose(0,1).reshape(h,72,64,128).index_select(1,tiles).index_select(2,offsets).flatten(1,2);own=(tiles+p-72)[None].expand(h,-1)
            def build(m):
                hs=SummaryOps.build(m,k[:,:p-72],v[:,:p-72]);cs=SummaryOps.build(m,k[:,p-72:],v[:,p-72:],cheap=True)
                return Summary(*(torch.cat((getattr(hs,n),getattr(cs,n)),1) for n in ('keys','mean','detail','stats')))
            summaries={name:build(model[l]) for name,model in models.items()}
            def route(name):
                f,_=RouteOps.features(qs,summaries[name]);score=(f*models[name][l].theta[:,None,None]).sum(-1).reshape(h,2,16,p).mean(2)
                return RouteOps.select(score,own,Config())['mask']
            finalmask=route('final');oldids=result[0][0].index_select(1,tiles).long();oldmask=torch.zeros_like(finalmask).scatter(-1,oldids,True)
            expand=lambda mask:mask[:,:,None].expand(-1,-1,16,-1).reshape(h,32,p)
            target,zd=AttentionOps.evaluate(qs,k,v,summaries['final'],models['final'][l].basis,torch.ones(h,32,p,device=q.device,dtype=torch.bool));rows={}
            def evaluate(name,s,m,mask):
                assert torch.equal(mask.sum(-1),finalmask.sum(-1))
                out,z=AttentionOps.evaluate(qs,k,v,s,m.basis,expand(mask));error=(out-target).square().sum();energy=target.square().sum()
                rows[name]=dict(nmse=float(error/energy),squared_error=float(error),target_energy=float(energy),log_denominator_mae=float((z-zd).abs().mean()),selected=int(mask.sum()),mask_difference=int((mask!=finalmask).sum()))
            m=models['final'][l];s=summaries['final'];evaluate('full',s,m,finalmask)
            for variant in ('projected_mean','no_detail'):evaluate(variant,change_summary(s,m.basis,rt.model[l].center,variant),m,finalmask)
            evaluate('old_route',s,m,oldmask)
            mass=RouteOps.select(mass_scores(qs,s,2),own,Config())['mask'];evaluate('mass_route',s,m,mass)
            for name in ('initial','pretrained'):
                evaluate(name+'_fixed_mask',summaries[name],models[name][l],finalmask)
                evaluate(name+'_own_route',summaries[name],models[name][l],route(name))
            exact=torch.nn.functional.scaled_dot_product_attention(qs.float(),k.flatten(1,2).float(),v.flatten(1,2).float(),attn_mask=expand(finalmask).repeat_interleave(64,-1))
            error=(exact-target).square().sum();rows['no_tail']=dict(nmse=float(error/target.square().sum()),squared_error=float(error),target_energy=float(target.square().sum()),selected=int(finalmask.sum()),mask_difference=0)
            atomic(dest,dict(chunk=chunk,layer=l,step=step,pages=p,rows=rows));print(json.dumps(dict(chunk=chunk,layer=l,step=step,rows={n:x['nmse'] for n,x in rows.items()})),flush=True)
            return result
        rt.decision=decision;emb=torch.load(DATA/'embeddings'/f'{a.case}.pt',weights_only=True).cuda()
        for chunk in range(9):generate_chunk(engine,emb,101,chunk)
        assert len(list(root.glob('c*_l*_s*.json')))==36
        atomic(root/'complete.json',dict(status='complete',events=36,identity=identity))
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
