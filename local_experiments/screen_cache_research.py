"""Evaluate cached grouping and output-aware attention on real dense trajectories."""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import torch
from .cache_grouping import PAGE,CHUNK,history_order,reorder,page_stats,page_budget

GROUPS={'spatial':None,'k':0.,'kv025':.25,'kv1':1.}
MODES=['mass','value','tail','var_tail','hysteresis','oracle_mass']


def timed(fn,repeats=1):
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(repeats):result=fn()
    torch.cuda.synchronize()
    return result,(time.perf_counter()-start)*1000/repeats


def evaluate(q,route,kh,vh,kc,vc,stats,mode,ratio=.2,prior=None):
    h,n,d=kh.shape;p=n//PAGE;scale=d**-.5
    km,vm,kvar,vvar=stats
    lh=(q@kh.float().transpose(1,2))*scale
    lc=(q@kc.float().transpose(1,2))*scale
    maximum=torch.maximum(lh.amax(-1),lc.amax(-1))
    ah=(lh-maximum[:,:,None]).exp().reshape(h,q.shape[1],p,PAGE)
    ac=(lc-maximum[:,:,None]).exp()
    z=ah.sum(-1)
    u=torch.einsum('hqpn,hpnd->hqpd',ah,vh.float().reshape(h,p,PAGE,d))
    zc=ac.sum(-1);uc=ac@vc.float()
    ref=(u.sum(2)+uc)/(z.sum(2)+zc)[:,:,None]
    logits=route@km.transpose(1,2)*scale
    if mode=='hysteresis' and prior is not None:logits=logits+.25*prior
    if mode=='value':
        # Pilot output from mean KV of all pages, including the current chunk.
        ckm,cvm,_,_=page_stats(kc,vc)
        all_logits=torch.cat((logits,route@ckm.transpose(1,2)*scale),-1)
        pilot=all_logits.softmax(-1)@torch.cat((vm,cvm),1)
        residual=(vm[:,None]-pilot[:,:,None]).square().sum(-1)+vvar[:,None]
        logits=logits+.5*residual.clamp_min(1e-12).log()
    if mode=='var_tail':logits=logits+.5*(route.square()@kvar.transpose(1,2))/d
    if mode=='oracle_mass':logits=z.clamp_min(1e-30).log()
    tail=mode in ('tail','var_tail');budget=page_budget(p,ratio,tail,mode=='var_tail')
    mask=torch.zeros_like(z,dtype=torch.bool).scatter_(-1,logits.topk(budget,dim=-1).indices,True)
    true_tail=(z*(~mask)).sum(-1)/(zc+z.sum(-1))
    estimated_tail=torch.zeros_like(true_tail)
    # Normalize the actual retained set. A huge dropped logit or a huge
    # *selected* Gaussian estimate must not set the sparse stabilizer, or all
    # contributing terms can underflow to zero although the output is finite.
    retained_logits=lh.masked_fill(~mask.repeat_interleave(PAGE,dim=-1),-torch.inf)
    sparse_max=torch.maximum(retained_logits.amax(-1),lc.amax(-1))
    if tail:
        estimate=q@km.transpose(1,2)*scale+math.log(PAGE)
        if mode=='var_tail':estimate=estimate+.5*(q.square()@kvar.transpose(1,2))/d
        estimate=estimate.masked_fill(mask,-torch.inf)
        sparse_max=torch.maximum(sparse_max,estimate.amax(-1))
    retained_weights=(retained_logits-sparse_max[:,:,None]).exp()
    current_weights=(lc-sparse_max[:,:,None]).exp()
    num=retained_weights@vh.float()+current_weights@vc.float()
    den=retained_weights.sum(-1)+current_weights.sum(-1)
    if tail:
        az=(estimate-sparse_max[:,:,None]).exp()
        num=num+az@vm;den=den+az.sum(-1)
        estimated_tail=az.sum(-1)/den
    output=num/den[:,:,None]
    assert torch.isfinite(output).all(),(mode,'non-finite sparse diagnostic output')
    denom=ref.square().sum((1,2)).clamp_min(1e-20)
    metrics=dict(nmse=(output-ref).square().sum((1,2))/denom,
      output_mean_bias=(output-ref).mean(1).square().sum(-1)/(denom/q.shape[1]),
      retained_probability=(zc+(z*mask).sum(-1))/(zc+z.sum(-1)),
      history_mass_coverage=(z*mask).sum(-1)/z.sum(-1).clamp_min(1e-20))
    if tail:
        metrics.update(tail_mass_actual=true_tail.mean(-1),tail_mass_estimated=estimated_tail.mean(-1),
            tail_mass_mae=(estimated_tail-true_tail).abs().mean(-1),tail_mass_estimated_max=estimated_tail.amax(-1))
    for key in ('retained_probability','history_mass_coverage'):metrics[key]=metrics[key].mean(-1)
    return metrics,mask,output,ref,budget


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='artifacts/cache_research_20261029')
    a=parser.parse_args();root=Path(a.root);out=root/'screen';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=[];timings=[];checks=[]
    for case in ['sailboat','skateboard','dog']:
        for path in sorted((root/'capture'/case/'qkv').glob('*_history.pt')):
            data=torch.load(path,weights_only=True);kh=data['k'].cuda();vh=data['v'].cuda()
            h,n,d=kh.shape;stem=path.name.removesuffix('_history.pt')
            for group,weight in GROUPS.items():
                order,ms=timed(lambda:history_order(kh,vh,weight))
                assert torch.equal(order.sort(-1).values,torch.arange(n,device='cuda').expand(h,n))
                k=reorder(kh,order);v=reorder(vh,order);stats=page_stats(k,v)
                timings.append(dict(case=case,chunk=data['chunk'],layer=data['layer'],group=group,
                    preprocess_ms_five_chunks_three_heads=ms,permutation_bytes=order.numel()*4,
                    mean_kv_bytes_if_persisted=2*h*(n//PAGE)*d*2,
                    diagonal_variance_bytes_if_persisted=h*(n//PAGE)*d*4))
                previous={}
                for step in range(4):
                    current=torch.load(path.with_name(f'{stem}_s{step}.pt'),weights_only=True)
                    qfull=current['q'].cuda().float();kc=current['k'].cuda();vc=current['v'].cuda()
                    ids=torch.linspace(0,qfull.shape[1]-1,64,device='cuda').long()
                    q=qfull[:,ids];route=qfull.reshape(h,-1,PAGE,d).mean(2)[:,ids//PAGE]
                    for mode in MODES:
                        metrics,mask,pred,ref,budget=evaluate(q,route,k,v,kc,vc,stats,mode,prior=previous.get(mode))
                        churn=torch.zeros(h,device='cuda')
                        if mode in previous:
                            before=previous[mode];churn=1-(before&mask).sum((1,2))/(before|mask).sum((1,2))
                        previous[mode]=mask
                        for head in range(h):
                            row=dict(case=case,chunk=data['chunk'],layer=data['layer'],head=data['heads'][head],
                                step=step,timestep=current['timestep'],group=group,mode=mode,selected_pages=budget,
                                exact_history_tokens=budget*PAGE,summary_tokens=(n//PAGE)*(2 if mode=='var_tail' else 1) if 'tail' in mode else 0,
                                mask_churn=float(churn[head]) if step else None)
                            row.update({key:float(val[head]) for key,val in metrics.items()});rows.append(row)
                    if not checks:
                        m,mask,pred,ref,b=evaluate(q,route,k,v,kc,vc,stats,'mass',1.)
                        assert float(m['nmse'].max())<1e-10
                        shift=torch.arange(d,device='cuda').float()[None,None]/128
                        old=evaluate(q,route,k,v,kc,vc,stats,'value')
                        new=evaluate(q,route,k,v.float()+shift,kc,vc.float()+shift,page_stats(k,v.float()+shift),'value')
                        assert torch.equal(old[1],new[1])
                        shift_error=float((new[2]-old[2]-shift).abs().max())
                        assert shift_error<1e-4,shift_error
                        checks.append(dict(full_budget_nmse=float(m['nmse'].max()),value_shift_max_error=shift_error,
                                           fixed_page_size=64,mask_invariant_to_value_shift=True))
                print(json.dumps(dict(case=case,capture=stem,group=group,preprocess_ms=ms,rows=len(rows))),flush=True)
            # Save incrementally so an interrupted screen can still be inspected.
            with (out/'metrics.csv').open('w') as f:
                fields=list(dict.fromkeys(key for row in rows for key in row))
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            (out/'timing.json').write_text(json.dumps(timings,indent=2)+'\n')
            (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    summary=[]
    for group in GROUPS:
        for mode in MODES:
            sub=[r for r in rows if r['group']==group and r['mode']==mode]
            summary.append(dict(group=group,mode=mode,n=len(sub),
                **{key:sum(r[key] for r in sub)/len(sub) for key in ['nmse','output_mean_bias','retained_probability','history_mass_coverage']}))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
