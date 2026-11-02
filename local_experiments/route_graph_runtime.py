"""Exact eager arithmetic under shared CUDA graphs; graph KV inputs budgeted."""
import math
import torch
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_learn_kernels import pool
from .high_sparse_learn import swap_features
from .high_sparse_gain import score

def postprocess(raw,code,vvar,kvar,w,p,keep,enabled):
    logits=(raw/math.sqrt(128)).reshape(12,72,4,p,4)
    logz=logits.logsumexp(-1)+math.log(16);prob=logz.softmax(-1);mass=prob.mean(2)
    own=(torch.arange(72,device=raw.device)+p-72)[None].expand(12,-1)
    z=mass.clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    if not enabled:return ids.sort(-1).values[None].int(),None
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    lm=mass.clamp_min(1e-12).log();lp=prob.clamp_min(1e-12).log()
    peak=(prob.max(2).values.clamp_min(1e-12).log()-lm).clamp(0,10)/5
    spread=lp.std(2,unbiased=False).clamp(0,5)/5
    current=(torch.arange(p,device=raw.device)>=p-72).float()[None,None].expand(12,72,-1)
    vv=vvar.float().clamp_min(0).log1p()[:,None].expand(-1,72,-1)
    kv=kvar[:,None].expand(-1,72,-1);lc=lm.clamp(-20,0)/5
    feat=torch.stack((torch.ones_like(lm),lc,peak,spread,vv,kv,current,current*lc),-1)
    sf=swap_features(feat,mass,code.float(),mask);s=score(mass,sf,w)
    rem=s.masked_fill(~mask,float('inf')).clone();rem.scatter_(-1,own[...,None],float('inf'))
    low,drop=rem.min(-1);high,add=s.masked_fill(mask,-float('inf')).max(-1);change=high>low
    lut=torch.where((ids==drop[...,None])&change[...,None],add[...,None],ids).sort(-1).values[None].int()
    return lut,(mass,sf,mask,own,add,drop)

class RouteGraph:
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
                for _ in range(3):warm=postprocess(*args)
            torch.cuda.current_stream().wait_stream(stream);del warm
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=postprocess(*args)
            self.graphs[key]=g;self.outputs[key]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs[key].replay()
        return self.outputs[key]

class GraphRuntime(GainRuntime):
    def __init__(self,engine,checkpoint,strength=1.):
        assert strength in [0.,1.]
        super().__init__(engine,checkpoint,strength)
        if not hasattr(engine,'shared_route_graph'):engine.shared_route_graph=RouteGraph()
        self.router=engine.shared_route_graph;self.persistent_bytes+=self.router.kv_aux_bytes
        assert self.persistent_bytes<=self.budget,(self.persistent_bytes,self.budget)
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
    def finish(self):return super().finish()|dict(route_implementation='eager operations replayed in CUDA graph; direct LUT update; probes gathered before fp32 conversion',graph_kv_input_bytes=self.router.kv_aux_bytes,graph_count=len(self.router.graphs),graph_capture_seconds=self.router.capture_seconds,graph_working_storage_note='score/probability/features/CUDA graph intermediates are computation workspace, not retained historical KV; peak GPU memory reported separately')
