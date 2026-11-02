"""Train-query response geometry, strictly hard balanced pages and prototypes."""
import argparse
import json
import math
import signal
from pathlib import Path
import torch
from .cache_grouping import history_order, reorder
from .learned_cache.data import records, teacher
from .learned_cache.model import predict, initial_mask, output_for_masks
from .learned_cache.run import nmse
from .learned_cache.io import atomic, digest, disk_guard, RunLock

METHODS=['k','response','response_v025','response_v1','random_response']

def normalize(x):
    x=x-x.mean(1,keepdim=True)
    return x/x.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)

def balanced(x,factors):
    h,n,d=x.shape;ids=torch.arange(n,device=x.device).expand(h,n).clone();groups=1
    for factor in factors:
        size=n//groups;x=x.reshape(h*groups,size,d);ids=ids.reshape(h*groups,size)
        xc=x-x.mean(1,keepdim=True);var=xc.square().mean(1)
        direction=torch.zeros_like(var).scatter_(1,var.argmax(-1,keepdim=True),1.)
        for _ in range(4):
            direction=torch.bmm(xc.transpose(1,2),torch.bmm(xc,direction.unsqueeze(-1))).squeeze(-1)
            direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        order=torch.bmm(xc,direction.unsqueeze(-1)).squeeze(-1).argsort(dim=-1,stable=True)
        x=x.gather(1,order[:,:,None].expand_as(x));ids=ids.gather(1,order);groups*=factor
    return ids.reshape(h,n)

@torch.no_grad()
def dictionary():
    result={}
    for layer in [4,14,24]:
        samples=[]
        for path in records('train'):
            if f'l{layer:02d}' not in path.name:continue
            q=torch.load(path,weights_only=True,map_location='cpu')['q'].float().cuda()
            ids=torch.linspace(0,q.shape[1]-1,128,device='cuda').long();samples.append(q[:,ids])
        q=torch.cat(samples,1);h,n,d=q.shape
        # Select actual observed queries, retaining their norms; no validation/test Q.
        chosen=[];distance=torch.full((h,n),float('inf'),device='cuda')
        index=(q-q.mean(1,keepdim=True)).square().sum(-1).argmax(-1)
        for _ in range(32):
            center=q[torch.arange(h,device='cuda'),index];chosen.append(center)
            distance=torch.minimum(distance,(q-center[:,None]).square().sum(-1));index=distance.argmax(-1)
        result[layer]=torch.stack(chosen,1).cpu()
    return result

def features(k,v,queries,method):
    if method=='k':return normalize(k)
    x=normalize((k-k.mean(1,keepdim=True))@queries.transpose(1,2)/math.sqrt(k.shape[-1]))
    if method.startswith('response_v'):
        weight=.25 if method.endswith('025') else 1.
        x=torch.cat((x,normalize(v)*math.sqrt(weight)),-1)
    return x

def grouping(k,v,queries,method):
    orders=[];fine=[]
    for start in range(0,k.shape[1],4608):
        x=features(k[:,start:start+4608],v[:,start:start+4608],queries,method)
        order=balanced(x,[3,3,2,2,2]);orders.append(order+start)
        xp=reorder(x,order).reshape(-1,64,x.shape[-1])
        inner=balanced(xp,[2,2]).reshape(k.shape[0],-1,64)
        fine.append(inner)
    order=torch.cat(orders,1);inner=torch.cat(fine,1)
    assert torch.equal(order.sort(-1).values,torch.arange(k.shape[1],device=k.device).expand_as(order))
    assert torch.equal(order//4608,torch.arange(k.shape[1],device=k.device).expand_as(order)//4608)
    assert torch.equal(inner.sort(-1).values,torch.arange(64,device=k.device).expand_as(inner))
    kr=reorder(k,order);vr=reorder(v,order);h,n,d=k.shape
    kp=kr.reshape(h,-1,64,d);vp=vr.reshape_as(kp)
    pk=kp.gather(2,inner[:,:,:,None].expand_as(kp)).reshape(h,-1,4,16,d).mean(3)
    pv=vp.gather(2,inner[:,:,:,None].expand_as(vp)).reshape(h,-1,4,16,d).mean(3)
    proto4=(pk,pv,torch.full(pk.shape[:3],16.,device=k.device))
    proto1=(kp.mean(2)[:,:,None],vp.mean(2)[:,:,None],torch.full((*kp.shape[:2],1),64.,device=k.device))
    return kr,vr,proto1,proto4,order

@torch.no_grad()
def run(args):
    root=Path(args.root);lock=RunLock(root);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    sources=[Path(__file__),Path('local_experiments/cache_grouping.py')]+list(Path('local_experiments/learned_cache').glob('*.py'))
    data=sorted(set(records('train')+records('val')+records('test')))
    data=sorted(set(data+[p.with_name(p.name[:-6]+'_history.pt') for p in data]))
    identity=dict(code={str(p):digest(p) for p in sources},data={str(p):digest(p) for p in data},methods=METHODS,queries=64,dictionary_size=32)
    try:
        if (root/'identity.json').exists():
            assert json.loads((root/'identity.json').read_text())==identity,'Resume identity mismatch'
        else:atomic(root/'identity.json',identity)
        disk_guard(root);atomic(root/'status.json',dict(status='running'))
        if (root/'dictionary.pt').exists():bank=torch.load(root/'dictionary.pt',weights_only=True)
        else:bank=dictionary();atomic(root/'dictionary.pt',bank,True)
        bank={k:v.cuda() for k,v in bank.items()};random_bank={}
        generator=torch.Generator(device='cuda').manual_seed(123)
        for layer,q in bank.items():
            r=torch.randn(q.shape,device='cuda',generator=generator)
            random_bank[layer]=r/r.norm(dim=-1,keepdim=True)*q.norm(dim=-1,keepdim=True)
        done=0;reused=0;cached_path=None;cached=None
        for split in ['val','test']:
            for path in records(split):
                out=root/split/(path.parent.parent.name+'_'+path.stem+'.json')
                if out.exists():reused+=1;continue
                if stop[0]:break
                disk_guard(root)
                hp=path.with_name(path.name[:-6]+'_history.pt')
                if hp!=cached_path:
                    hist=torch.load(hp,weights_only=True,map_location='cpu');k=hist['k'].cuda().float();v=hist['v'].cuda().float();layer=hist['layer'];cached={}
                    for method in METHODS:
                        qbank=random_bank[layer] if method=='random_response' else bank[layer]
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        a.record();group=grouping(k,v,qbank,method);b.record();b.synchronize()
                        cached[method]=(group,a.elapsed_time(b))
                    assert torch.equal(cached['k'][0][-1],history_order(k,v,0.)),'K baseline changed'
                    cached_path=hp
                event=torch.load(path,weights_only=True,map_location='cpu');q=event['q'].cuda().float();kc=event['k'].cuda().float();vc=event['v'].cuda().float()
                ids=torch.linspace(0,q.shape[1]-1,64,device='cuda').long();route=q.reshape(3,-1,64,128).mean(2)[:,ids//64];q=q[:,ids]
                rows=[];reference=None;max_delta=0.
                for method in METHODS:
                    (kr,vr,p1,p4,_),ms=cached[method]
                    z,values,target=teacher(q,kr,vr,kc,vc)
                    if reference is None:reference=target
                    else:
                        delta=(target-reference).abs().max().item();max_delta=max(max_delta,delta)
                        assert torch.allclose(target,reference,atol=2e-5,rtol=2e-4),delta
                    for count,proto in [(1,p1),(4,p4)]:
                        rz,_,_=predict(route,proto,kc,vc);mask=initial_mask(rz)
                        assert (mask.sum(-1)==73).all() and mask[...,-1].all()
                        sparse=output_for_masks(z,values,mask[:,:,None])[:,:,0]
                        _,_,approx=predict(q,proto,kc,vc)
                        retained=(z.softmax(-1)*mask).sum(-1)
                        error=nmse(sparse,reference);fit=nmse(approx,reference)
                        for hi,head in enumerate(hist['heads']):
                            rows.append(dict(split=split,scene=path.parent.parent.name,chunk=event['chunk'],layer=hist['layer'],head=int(head),step=int(path.stem[-1]),
                                grouping=method,prototypes=count,nmse=error[hi].mean().item(),summary_nmse=fit[hi].mean().item(),retained_mass=retained[hi].mean().item(),preprocess_5chunks_ms=ms))
                atomic(out,dict(rows=rows,dense_permutation_max_abs=max_delta));done+=1
                atomic(root/'status.json',dict(status='running',new_events=done,reused_events=reused))
                print(out.name,flush=True)
                if args.max_events and done>=args.max_events:stop[0]=True
            if stop[0]:break
        allrows=[];max_delta=0.
        for p in sorted(root.glob('*/*.json')):
            obj=json.loads(p.read_text());allrows.extend(obj['rows']);max_delta=max(max_delta,obj['dense_permutation_max_abs'])
        summary={}
        for split in ['val','test']:
            summary[split]={}
            for method in METHODS:
                for count in [1,4]:
                    rows=[r for r in allrows if r['split']==split and r['grouping']==method and r['prototypes']==count]
                    if rows:summary[split][f'{method}_p{count}']={key:sum(r[key] for r in rows)/len(rows) for key in ['nmse','summary_nmse','retained_mass']}
        atomic(root/'summary.json',summary)
        atomic(root/'status.json',dict(status='interrupted' if stop[0] else 'completed',new_events=done,reused_events=reused,
             dense_permutation_max_abs=max_delta,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,disk=disk_guard(root)))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        atomic(root/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/response_hard_20261030');p.add_argument('--max-events',type=int,default=0)
    run(p.parse_args())
