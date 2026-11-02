"""Research-only cached grouping + optional second-moment tail compensation.

Persistent cache stores only integer permutations, no duplicate/summary K/V.
Means/variances are transient recomputed tensors in this first prototype.
LF's unchanged sparse forward kernel supplies exact selected output and LSE.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch
from .cache_grouping import PAGE,CHUNK,balanced_order,reorder,page_budget


def load_kernel():
    name='research_lightforcing_kernel'
    if name not in sys.modules:
        path=Path(__file__).resolve().parents[1]/'external_baselines/LightForcing/wan/modules/kernel.py'
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name]._attn_fwd


@torch.no_grad()
def sparse_forward(q,k,v,orders=None,tail=False,ratio=.2):
    """BLHD operands; all current KV dense, per-tile history page selection."""
    b,nq,h,d=q.shape;nh=k.shape[1]-nq
    assert b==1 and nq%PAGE==0 and nh%PAGE==0
    p=nh//PAGE;qm=nq//PAGE
    if orders is not None:
        current=torch.arange(nh,nh+nq,device=q.device).expand(h,nq)
        order=torch.cat((orders,current),-1)
        k=reorder(k[0].transpose(0,1),order).transpose(0,1).unsqueeze(0).contiguous()
        v=reorder(v[0].transpose(0,1),order).transpose(0,1).unsqueeze(0).contiguous()
    else:k=k.contiguous();v=v.contiguous()
    q=q.contiguous()
    kp=k[0,:nh].transpose(0,1).float().reshape(h,p,PAGE,d)
    km=kp.mean(2)
    query=q[0].transpose(0,1).float()
    route=query.reshape(h,qm,PAGE,d).mean(2)
    score=route@km.transpose(1,2)*d**-.5
    if tail:
        kvar=(kp-km[:,:,None]).square().mean(2)
        score=score+.5*(route.square()@kvar.transpose(1,2))/d
    keep=page_budget(p,ratio,tail,tail)
    chosen=score.topk(keep,dim=-1).indices
    always=torch.arange(p,p+qm,device=q.device).expand(h,qm,qm)
    lut=torch.cat((chosen,always),-1).sort(-1).values.unsqueeze(0).contiguous()
    output=torch.empty_like(q);lse=torch.empty((b,h,nq),device=q.device,dtype=torch.float32)
    load_kernel()[(qm,b*h)](q,k,v,d**-.5,lut.shape[-1],lut,lse,output,
                           h,nq,k.shape[1],qm,d,PAGE,PAGE,num_warps=4,num_stages=3)
    if tail:
        vm=v[0,:nh].transpose(0,1).float().reshape(h,p,PAGE,d).mean(2)
        selected=torch.zeros((h,qm,p),device=q.device,dtype=torch.bool).scatter_(-1,chosen,True)
        logits=query@km.transpose(1,2)*d**-.5+math.log(PAGE)
        logits+=.5*(query.square()@kvar.transpose(1,2))/d
        logits=logits.reshape(h,qm,PAGE,p).masked_fill(selected[:,:,None],-torch.inf).reshape(h,nq,p)
        tail_lse=torch.logsumexp(logits,dim=-1)
        tail_out=logits.softmax(-1)@vm
        mixing=(tail_lse-lse[0]*math.log(2)).sigmoid()
        exact=output[0].transpose(0,1).float()
        output=((1-mixing[:,:,None])*exact+mixing[:,:,None]*tail_out).transpose(0,1).unsqueeze(0).to(q.dtype)
    return output,dict(history_pages=p,selected_pages=keep,
        exact_history_tokens=keep*PAGE,summary_equivalent_tokens=2*p if tail else 0,
        current_tokens=nq,history_budget_tokens=int(nh*ratio))


class PrototypeAttention:
    def __init__(self,config,output,current,original):
        self.config=config;self.output=Path(output);self.current=current;self.original=original
        self.orders={i:{} for i in range(30)};self.rows=[];self.events=[];self.uses={}
        self.max_permutation_bytes=0;self.sparse_calls=0
        self.weight=config['weight'];self.tail=config.get('tail',False)

    @torch.no_grad()
    def __call__(self,q,k,v,*args,**kwargs):
        c=self.current;layer=c['layer'];chunk=c['chunk'];cache=self.orders[layer]
        for key in list(cache):
            if key<chunk-5:del cache[key]
        if c['timestep']==0 or layer==0 or chunk==0:
            out=self.original(q,k,v,*args,**kwargs)
            # Current K/V are overwritten by the clean pass. Only then freeze
            # grouping, keyed by absolute chunk id/layer until FIFO eviction.
            if c['timestep']==0 and layer>0 and self.weight is not None and chunk<40:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                cache[chunk]=balanced_order(k[0,-CHUNK:].transpose(0,1),v[0,-CHUNK:].transpose(0,1),self.weight).int()
                end.record();self.events.append((start,end));self.uses[(layer,chunk)]=0
                # Retain only the five clean chunks needed by the next call.
                for key in list(cache):
                    if key<chunk-4:del cache[key]
                self.max_permutation_bytes=max(self.max_permutation_bytes,
                    sum(t.numel()*t.element_size() for cc in self.orders.values() for t in cc.values()))
            return out
        nh=(k.shape[1]-q.shape[1])//CHUNK
        ids=list(range(chunk-nh,chunk));order=None
        if self.weight is not None:
            assert all(i in cache for i in ids),(layer,chunk,ids,list(cache))
            order=torch.cat([cache[i]+j*CHUNK for j,i in enumerate(ids)],1)
            for i in ids:self.uses[(layer,i)]+=1
        enabled_layers=self.config.get('tail_layers')
        use_tail=self.tail and (enabled_layers is None or layer in enabled_layers)
        out,budget=sparse_forward(q,k,v,order,use_tail,self.config.get('ratio',.2))
        self.sparse_calls+=1
        if layer==1 or (enabled_layers is not None and layer in enabled_layers):self.rows.append(dict(c,**budget))
        return out

    def finish(self,latent,inputs_path):
        old=torch.load(Path(inputs_path).parent/'dense.latents.pt',map_location='cpu',weights_only=True)
        torch.cuda.synchronize()
        result=dict(config=self.config,sparse_calls=self.sparse_calls,budget_trace=self.rows,
            persistent_auxiliary_permutation_peak_bytes=self.max_permutation_bytes,
            persistent_summary_or_duplicate_kv_bytes=0,
            preprocess_cuda_ms=sum(a.elapsed_time(b) for a,b in self.events),
            preprocess_calls=len(self.events),reuse_counts=list(self.uses.values()),
            first_chunk_identical=torch.equal(old[:,:3],latent.cpu()[:,:3]),
            latent_nmse=float((old-latent.cpu()).float().square().mean()/old.float().square().mean()),
            notes=['Prototype, not a reproduction of SlimAV or Light Forcing routing.',
              'Only integer grouping permutations persist. Derived mean KV/variance and reordered copies are transient.',
              'Clean-context passes and layer 0 are dense; no custom rebase.',
              'Tail charges two full-page-summary equivalents per page against the same 20% historical budget.',
              'No claim of speedup: Python routing/reorder and transient summaries are not fused.'])
        (self.output/'research.json').write_text(json.dumps(result,indent=2)+'\n')
        assert result['first_chunk_identical']
