"""Paired Q64 counterfactuals on identical pre-eviction teacher events."""
import sys,json
from pathlib import Path
import torch
from cells import arrange
here=Path(__file__).resolve().parent
repo=here.parents[1]
sys.path[:0]=[str(repo/'profiling/clean_forward'),str(repo/'reproduction/vbench_current')]
from clean_policy import attach
from scope import apply_selected
from quota_mask import quota_mask
from quota_pool import pool
from local_experiments.review_v1.ops import Summary,haar
from local_experiments.broad_common import Engine
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights

def terms(q,k,v,s,basis):
    # Q64, pages, 64 tokens. Use a common per-query logit offset.
    z=q@k.flatten(0,1).T/128**.5
    off=z.amax(1,keepdim=True)
    w=(z-off).exp().reshape(64,-1,64)
    exact=torch.einsum('qpt,ptd->qpd',w,v)
    z=q@s.keys.flatten(0,1).T/128**.5
    rep=16*(z-off).exp().reshape(64,-1,4)
    values=s.mean[:,None]+torch.einsum('ar,pak,dk->prd',haar(q),s.detail,basis)
    approx=torch.einsum('qpr,prd->qpd',rep,values)
    return torch.cat((exact,w.sum(-1,keepdim=True)),-1),torch.cat((approx,rep.sum(-1,keepdim=True)),-1)

def assess(ex,ap,mask,reader,fixed,risk,own,slots,total):
    target=ex.sum(1);target=target[:,:-1]/target[:,-1:]
    energy=target.square().sum()
    diff=ex-ap
    def err(x):return (x[...,:-1]/x[...,-1:]-target).square().sum((-1,-2))/energy
    def mix(m):return ap.sum(1)+diff[:,m].sum(1)
    start=mix(mask)
    rr,aa,ll=pool(risk[None,None],mask[None,None],own.reshape(1,1))
    r,a=rr.flatten(),aa.flatten()
    def best(r,a):
        val=err(start)
        for i,j in zip(r.split(64),a.split(64)):
            if i.numel():val=torch.minimum(val,err(start[None]+(diff[:,j]-diff[:,i]).transpose(0,1)).min())
        return val
    i,j=torch.where(ll.reshape(len(r),len(a)))
    ids=torch.arange(len(mask),device=qdevice)
    remove=mask&(ids>=24)&(ids!=own)
    i2,j2=torch.where(remove[:,None]&(~mask)[None]&((ids[:,None]>=len(mask)-72)==(ids[None]>=len(mask)-72)))
    data=dict(risk=err(start),reader=err(mix(reader)),fixed=err(mix(fixed)),pool=best(r[i],a[j]),legal=best(i2,j2),summary=err(ap.sum(1)),energy=energy)
    for dc,dh in [(-1,0),(1,0),(0,-1),(0,1)]:
        cur=int(slots)+dc;hist=int(total-slots)+dh
        if not 1<=cur<=72 or not 0<=hist<=len(mask)-96:continue
        m=quota_mask(risk[None,None].expand(1,72,-1).contiguous(),own.expand(1,72),slots.new_tensor([cur]),cur+hist)[0,0]
        data[f'c{dc}h{dh}']=err(mix(m))
    return {k:float(x) for k,x in data.items()}

qdevice='cuda'
@torch.no_grad()
def run():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cfg=config('intersection_fast',local_assets=repo);configure(cfg)
    e=Engine()
    for b in e.wrapper.model.blocks:b.self_attn.sink_size=1
    rt=make_runtime(e,*load_weights(cfg)[:4],cfg);rt.sparse_clean=True
    gen=attach(rt,e,10);apply_selected(rt,repo/'profiling/clean_forward')
    rt.batched_group_original=rt.batched_group
    results=[]
    def probe(l,step,q,k,v,out):
        c=e.current['chunk']
        target=e.dense(q,k,v)
        if c not in (2,4) or l not in (3,8) or step!=0:return target
        model=rt.model[l];chunks=[]
        for j in range(1,c):
            pos=rt.stores[l].token_ids[:,72*j:72*(j+1)].reshape(12,-1)
            idx=pos.argsort(1)
            def raw(t):return t[0,4608*j:4608*(j+1)].transpose(0,1).gather(1,idx[...,None].expand(-1,-1,128)).transpose(0,1)[None].contiguous()
            kr,vr=raw(k),raw(v)
            learned=(k[0,4608*j:4608*(j+1)].transpose(0,1).reshape(12,72,64,128),v[0,4608*j:4608*(j+1)].transpose(0,1).reshape(12,72,64,128),pos-j*4608)
            chunks.append((kr,vr,learned))
        tile=(seed+17*l+c)%72
        hp=72*c
        own=torch.arange(hp,hp+72,device='cuda').expand(12,-1)
        bg=rt.batch_graph
        probes=bg.select_probes(q[0].transpose(0,1).reshape(12,72,64,128),rt.offsets)
        masks={}
        token_masks={}
        for cell in ('NN','NG','GN','GG'):
            groups=[arrange(rt,cell,model,kr,vr,g) for kr,vr,g in chunks]
            def pages(t,i):return torch.cat([t[0,:4608].transpose(0,1).reshape(12,72,64,128),*[g[i] for g in groups],t[0,-4608:].transpose(0,1).reshape(12,72,64,128)],1)
            kg,vg=pages(k,0),pages(v,1)
            ss=[rt.slabs[l].slice(0,72),*[rt.summary_graph(model,g[0].contiguous(),g[1].contiguous()) for g in groups],rt.slabs[l].slice(hp,hp+72)]
            s=Summary(*(torch.cat([getattr(t,n) for t in ss],1).contiguous() for n in ('keys','mean','detail','stats')))
            en=rt.encode(l,s);heads=rt.correction_scope(l,step)[1]
            entry=bg.get(probes,s,en,model.theta,model.basis,own,True,heads=heads)
            bg.stage(entry,s,en,model.theta,model.basis)
            ids,reader,risk,_=bg.replay(entry,probes,own)
            mask=quota_mask(risk,own,bg.current_slots,bg.n)
            if cell[1]=='N':masks[cell[0]]=mask.clone()
            for h in (3,4,9):
                sh=Summary(*(getattr(s,n)[h].float() for n in ('keys','mean','detail','stats')))
                ex,ap=terms(q[0,64*tile:64*(tile+1),h].float(),kg[h].float(),vg[h].float(),sh,model.basis[h].bfloat16().float())
                row=assess(ex,ap,mask[h,tile],reader[h,tile],masks[cell[0]][h,tile],risk[h,tile],own[h,tile],bg.current_slots[h],bg.n)
                order=torch.cat([rt.stores[l].token_ids[h,:72].flatten(),*[g[2][h].flatten()+(j+1)*4608 for j,g in enumerate(groups)],torch.arange(hp*64,(hp+72)*64,device=q.device)])
                chosen=order[mask[h,tile].repeat_interleave(64)].sort().values
                if cell=='NN':token_masks[h]=chosen
                overlap=torch.isin(chosen,token_masks[h]).sum()
                row['token_jaccard_vs_NN']=float(overlap/(2*len(chosen)-overlap))
                row['selected_mass']=float((ex[:,mask[h,tile],-1].sum(1)/ex[:,:,-1].sum(1)).mean())
                if cell=='GG':
                    m=reader[h,tile]
                    pred=ex[:,m].sum(1)+ap[:,~m].sum(1)
                    pred=pred[:,:-1]/pred[:,-1:]
                    actual=out[0,64*tile:64*(tile+1),h].float()
                    row['replay_nmse']=float((pred-actual).square().sum()/actual.square().sum())
                results.append(dict(prompt=p,seed=seed,chunk=c,layer=l,head=h,tile=tile,cell=cell,**row))
        return target
    rt.sparse_output_probe=probe
    for p in ('city_walk','short_721'):
        path=repo/'profiling/clean_forward/embeddings/city_walk.pt' if p=='city_walk' else repo.parent/'.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt'
        emb=torch.load(path,weights_only=True).cuda()
        for seed in (721,722):
            rt.reset();gen(emb,seed,5)
            (here/'events.json').write_text(json.dumps(results))
run()
