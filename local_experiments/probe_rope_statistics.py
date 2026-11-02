"""Coordinate-invariance diagnostic, NOT a rebase video comparison.

The same orthogonal rotation is applied to Q and every K: exact attention must
stay fixed. Test whether diagonal vs pair-block cached moments respect this.
"""
import json
import math
from pathlib import Path
import torch
from .cache_grouping import history_order,reorder


def rotate(x):
    y=x.reshape(*x.shape[:-1],64,2).clone()
    # Wan head dim 128 assigns 22 complex pairs to the temporal component.
    angles=torch.linspace(-2.8,2.8,22,device=x.device)
    a,b=y[...,:22,0].clone(),y[...,:22,1].clone()
    y[...,:22,0]=a*angles.cos()-b*angles.sin()
    y[...,:22,1]=a*angles.sin()+b*angles.cos()
    return y.flatten(-2)


def scores(q,k,paired):
    h,n,d=k.shape;pages=k.reshape(h,n//64,64,d);mu=pages.mean(2);delta=pages-mu[:,:,None]
    var=delta.square().mean(2)
    result=q@mu.transpose(1,2)/d**.5+.5*(q.square()@var.transpose(1,2))/d
    if paired:
        cross=(delta[...,0::2]*delta[...,1::2]).mean(2)
        result+=(q[...,0::2]*q[...,1::2])@cross.transpose(1,2)/d
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/cache_research_20261029');folder=root/'capture/sailboat/qkv'
    records=[]
    for layer in [4,14,24]:
        old=torch.load(folder/f'c20_l{layer:02d}_history.pt',weights_only=True)
        cur=torch.load(folder/f'c20_l{layer:02d}_s2.pt',weights_only=True)
        k,v,q,kc,vc=[x.cuda().float() for x in [old['k'],old['v'],cur['q'],cur['k'],cur['v']]]
        order=history_order(k,v,0.);k=reorder(k,order);v=reorder(v,order)
        ids=torch.linspace(0,q.shape[1]-1,64,device='cuda').long();q=q[:,ids]
        kr,qr,kcr=rotate(k),rotate(q),rotate(kc)
        allv=torch.cat((v,vc),1)
        exact=(q@torch.cat((k,kc),1).transpose(1,2)/128**.5).softmax(-1)@allv
        exactr=(qr@torch.cat((kr,kcr),1).transpose(1,2)/128**.5).softmax(-1)@allv
        nmse=float((exactr-exact).square().sum()/exact.square().sum())
        assert nmse<1e-10,nmse
        row=dict(layer=layer,exact_attention_rotation_nmse=nmse)
        for paired in [False,True]:
            before=scores(q,k,paired);after=scores(qr,kr,paired)
            delta=(after-before).abs();name='pair_blocks' if paired else 'diagonal'
            row[name+'_mean_log_mass_difference']=float(delta.mean())
            row[name+'_max_log_mass_difference']=float(delta.max())
            bm=torch.zeros_like(before,dtype=torch.bool).scatter_(-1,before.topk(60,dim=-1).indices,True)
            am=torch.zeros_like(after,dtype=torch.bool).scatter_(-1,after.topk(60,dim=-1).indices,True)
            row[name+'_mask_churn']=float(1-(bm&am).sum()/(bm|am).sum())
            if paired:assert float(delta.max())<1e-3
        records.append(row)
    result=dict(records=records,
        notes=['Real QKV, FP32 diagnostic, fixed grouping, identical global rotation of Q and K; exact attention unchanged.',
          'Not anchor-only rebase, not a video-quality result. Diagonal Gaussian moments are coordinate dependent.',
          '2x2 RoPE-pair covariance preserves this invariance, but does not solve non-Gaussian tails or missing K-V cross moments.'])
    (root/'rope_statistics_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
