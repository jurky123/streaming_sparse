"""All-head per-layer routers and on-the-fly grouped teacher statistics."""
import torch
from torch import nn
from .broad_common import ROOT,CHUNKS,cases,guard
from .learned_cache.io import atomic,digest
from .response_hard import features,balanced
from .cache_grouping import reorder
from .learned_cache.data import teacher
from .learned_cache.model import predict,initial_mask,output_for_masks
from .learned_cache.run import nmse


@torch.no_grad()
def dictionary():
    dest=ROOT/'dictionary.pt'
    if dest.exists():return
    bank={};sources={}
    for layer in range(30):
        samples=[]
        for case in cases('train'):
            for chunk in CHUNKS:
                path=ROOT/'trajectories'/case['id']/f'chunk{chunk:02d}.pt'
                data=torch.load(path,weights_only=True);q=data['queries'][:,layer].permute(1,0,2,3).flatten(1,2).cuda().float()
                samples.append(q)
                if layer==0:sources[str(path)]=digest(path)
        q=torch.cat(samples,1);heads,n,d=q.shape;distance=torch.full((heads,n),float('inf'),device='cuda')
        index=(q-q.mean(1,keepdim=True)).square().sum(-1).argmax(-1);selected=[]
        for _ in range(32):
            center=q[torch.arange(heads,device='cuda'),index];selected.append(center)
            distance=torch.minimum(distance,(q-center[:,None]).square().sum(-1));index=distance.argmax(-1)
        bank[layer]=torch.stack(selected,1).cpu();print('dictionary layer',layer,flush=True)
    atomic(dest,bank,True);atomic(ROOT/'dictionary_manifest.json',dict(source_hashes=sources,split='train',heads=12,layers=30,queries_per_head=32))


class LayerRouter(nn.Module):
    def __init__(self,layer):
        super().__init__();generator=torch.Generator().manual_seed(layer)
        self.weight=nn.Parameter(torch.randn(12,4,128,generator=generator)*.01)
        self.bias=nn.Parameter(torch.zeros(12,4))

    def forward(self,h):
        k=h['k'].reshape(12,-1,64,128);v=h['v'].reshape_as(k)
        a=(torch.einsum('hbtd,hrd->hbtr',k,self.weight)+self.bias[:,None,None]).softmax(-1)
        count=a.sum(2)
        return (torch.einsum('hbtr,hbtd->hbrd',a,k)/count.clamp_min(1e-20)[...,None],
                torch.einsum('hbtr,hbtd->hbrd',a,v)/count.clamp_min(1e-20)[...,None],count)


class Events:
    def __init__(self,layers):
        self.layers=set(layers);self.bank={k:v.cuda() for k,v in torch.load(ROOT/'dictionary.pt',weights_only=True).items() if k in self.layers}
        self.chunk=None;self.history={}

    def reset(self):self.chunk=None;self.history.clear()

    @torch.no_grad()
    def event(self,current,q,k,v):
        chunk=current['chunk'];layer=current['layer']
        if chunk!=self.chunk:self.history.clear();self.chunk=chunk
        heads=q.shape[2];n=q.shape[1];nh=k.shape[1]-n
        assert heads==12 and nh==23040 and n==4608
        if layer not in self.history:
            kh=k[0,:nh].transpose(0,1).float();vh=v[0,:nh].transpose(0,1).float();orders=[]
            for start in range(0,nh,4608):
                x=features(kh[:,start:start+4608],vh[:,start:start+4608],self.bank[layer],'response')
                order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
                inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(heads,-1,64)
                order=order.reshape(heads,-1,64).gather(2,inner).flatten(1);orders.append(order+start)
            order=torch.cat(orders,1);kh=reorder(kh,order);vh=reorder(vh,order)
            self.history[layer]=dict(k=kh,v=vh,mean=(kh.reshape(heads,-1,4,16,128).mean(3),
                vh.reshape(heads,-1,4,16,128).mean(3),torch.full((heads,nh//64,4),16.,device='cuda')))
        h=self.history[layer];fullq=q[0].transpose(0,1).float();ids=torch.linspace(0,n-1,32,device='cuda').long()
        query=fullq[:,ids];route=fullq.reshape(heads,-1,64,128).mean(2)[:,ids//64]
        kc=k[0,nh:].transpose(0,1).float();vc=v[0,nh:].transpose(0,1).float()
        z,values,target=teacher(query,h['k'],h['v'],kc,vc)
        return dict(h=h,q=query,route=route,kc=kc,vc=vc,z=z,values=values,target=target)


def prediction(model,e,route=False):
    return predict(e['route' if route else 'q'],model(e['h']),e['kc'],e['vc'])


@torch.no_grad()
def evaluate_event(model,e):
    proto=e['h']['mean'] if model is None else model(e['h'])
    z,_,_=predict(e['route'],proto,e['kc'],e['vc']);mask=initial_mask(z)
    assert (mask.sum(-1)==73).all() and mask[...,-1].all()
    out=output_for_masks(e['z'],e['values'],mask[:,:,None])[:,:,0]
    mse=(out-e['target']).square().sum((1,2));energy=e['target'].square().sum((1,2))
    return dict(nmse=nmse(out,e['target']).mean(-1).cpu().tolist(),sse=mse.cpu().tolist(),energy=energy.cpu().tolist(),
        max_query_nmse=nmse(out,e['target']).max(-1).values.cpu().tolist())
