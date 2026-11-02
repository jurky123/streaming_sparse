"""Fused inference router; historical arithmetic and initial top-k remain unchanged."""
import math
import torch
from .route_graph_runtime import GraphRuntime
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_learn_kernels import pool
from .route_fused_kernels import score_lut,summary

def fused_postprocess(raw,code,vvar,kvar,w,p,keep,enabled):
    logits=(raw/math.sqrt(128)).reshape(12,72,4,p,4)
    logz=logits.logsumexp(-1)+math.log(16);prob=logz.softmax(-1);mass=prob.mean(2)
    own=(torch.arange(72,device=raw.device)+p-72)[None].expand(12,-1)
    z=mass.clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    if not enabled:return ids.sort(-1).values[None].int(),None
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=weighted@code.float()/den;allvalue=mass@code.float()
    return score_lut(prob,mass,code,vvar,kvar,w,mask,den,selected,allvalue,ids),None

class FusedRouter:
    def __init__(self):
        self.raw=torch.empty(12*72*4*410*4,device='cuda')
        self.code=torch.empty(12*410*8,device='cuda',dtype=torch.bfloat16)
        self.vv=torch.empty(12*410,device='cuda');self.kvar=torch.empty_like(self.vv)
        self.weight=torch.empty(12,14,device='cuda');self.probes=torch.tensor([0,21,42,63],device='cuda')
        self.graphs={};self.outputs={};self.capture_seconds=0.
        self.kv_aux_bytes=sum(t.numel()*t.element_size() for t in [self.code,self.vv,self.kvar,self.weight,self.probes])
    @torch.no_grad()
    def __call__(self,q,pk,code,vvar,w,enabled):
        p=pk.shape[1];keep=int(math.floor(p*.1+1e-6));key=(p,bool(enabled))
        query=q[0].reshape(72,64,12,128).index_select(1,self.probes).permute(2,0,1,3).contiguous().float().reshape(12,288,128)
        raw=self.raw[:12*288*p*4].view(12,288,p*4)
        torch.bmm(query,pk.flatten(1,2).transpose(1,2),out=raw)
        kvar=self.kvar[:12*p].view(12,p)
        torch.log1p((pk-pk.mean(2,keepdim=True)).square().mean((2,3)),out=kvar)
        args=(self.raw[:raw.numel()].view_as(raw),self.code[:code.numel()].view_as(code),self.vv[:vvar.numel()].view_as(vvar),self.kvar[:kvar.numel()].view_as(kvar),self.weight,p,keep,bool(enabled))
        args[0].copy_(raw);args[1].copy_(code);args[2].copy_(vvar);args[3].copy_(kvar);args[4].copy_(w)
        if key not in self.graphs:
            import time
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=fused_postprocess(*args)
            torch.cuda.current_stream().wait_stream(stream);del warm
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=fused_postprocess(*args)
            self.graphs[key]=g;self.outputs[key]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs[key].replay()
        return self.outputs[key]


class FusedRuntime(GraphRuntime):
    def __init__(self,engine,checkpoint,strength=1.):
        GainRuntime.__init__(self,engine,checkpoint,strength)
        if not hasattr(engine,'shared_fused_router'):engine.shared_fused_router=FusedRouter()
        self.router=engine.shared_fused_router
        self.persistent_bytes+=self.router.kv_aux_bytes
        assert self.persistent_bytes<=self.budget
        self.route_meta=None
    def route(self,l,q,p):
        cache=self.engine.cache[l];device=q.device
        ck=cache['k'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,4,16,128).mean(3)
        vm=cache['v'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,64,128).mean(2)
        cc=((vm-self.centers[l][:,None])@self.basis[l].float()).to(torch.bfloat16)
        var=torch.empty(12,72,device=device,dtype=torch.bfloat16)
        pool(cache,self.basis[l],self.centers[l],self.uk[l],self.uv[l],cc,var,offset=p*64,weighted=False)
        self.active_pk=torch.cat((self.pk[l][:,:p],ck),1).contiguous();self.active_codes=self.router.code[:12*(p+72)*8].view(12,p+72,8)
        torch.cat((self.codes[l][:,:p],cc),1,out=self.active_codes)
        vv=self.router.vv[:12*(p+72)].view(12,p+72)
        torch.cat((self.vvar[l][:,:p],var),1,out=vv)
        lut,meta=self.router(q,self.active_pk,self.active_codes,vv,self.sw[l],bool(self.strength));self.route_meta=meta
        keep=lut.shape[-1];self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=keep,exact_sparsity=1-keep/(p+72))
        return lut
    def finish(self):
        return super().finish()|dict(route_implementation='native mass/topk/bmm, fused Triton features-score and swap-LUT; inference only')
