"""Capture QK and native K variance as well; share computed packed-K workspace."""
import math,time
import torch
from .route_fused_runtime import FusedRouter,FusedRuntime,fused_postprocess
from .route_fused_kernels import gather
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_learn_kernels import pool
class PackedRouter(FusedRouter):
    postprocess=staticmethod(fused_postprocess)
    def __init__(self):
        super().__init__()
        self.query=torch.empty(12,288,128,device='cuda')
        self.packed_pk=torch.empty(12*410*4*128,device='cuda')
    @torch.no_grad()
    def __call__(self,q,pk,code,vvar,w,enabled):
        p=pk.shape[1];key=(p,bool(enabled));keep=int(math.floor(p*.1+1e-6))
        packed=self.packed_pk[:pk.numel()].view_as(pk)
        if packed.data_ptr()!=pk.data_ptr():packed.copy_(pk)
        gather(q,self.query)
        c=self.code[:code.numel()].view_as(code);vv=self.vv[:vvar.numel()].view_as(vvar)
        c.copy_(code);vv.copy_(vvar);self.weight.copy_(w)
        def pipeline():
            raw=self.raw[:12*288*p*4].view(12,288,p*4)
            torch.bmm(self.query,packed.flatten(1,2).transpose(1,2),out=raw)
            kv=self.kvar[:12*p].view(12,p)
            torch.log1p((packed-packed.mean(2,keepdim=True)).square().mean((2,3)),out=kv)
            return self.postprocess(raw,c,vv,kv,self.weight,p,keep,bool(enabled))
        if key not in self.graphs:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=pipeline()
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):out=pipeline()
            self.graphs[key]=graph;self.outputs[key]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs[key].replay();return self.outputs[key]
class PackedRuntime(FusedRuntime):
    def __init__(self,engine,checkpoint,strength=1.):
        GainRuntime.__init__(self,engine,checkpoint,strength)
        if not hasattr(engine,'shared_packed_router'):engine.shared_packed_router=PackedRouter()
        self.router=engine.shared_packed_router;self.persistent_bytes+=self.router.kv_aux_bytes
        assert self.persistent_bytes<=self.budget;self.route_meta=None
    def route(self,l,q,p):
        cache=self.engine.cache[l];device=q.device
        ck=cache['k'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,4,16,128).mean(3)
        vm=cache['v'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,64,128).mean(2)
        cc=((vm-self.centers[l][:,None])@self.basis[l].float()).to(torch.bfloat16)
        var=torch.empty(12,72,device=device,dtype=torch.bfloat16)
        pool(cache,self.basis[l],self.centers[l],self.uk[l],self.uv[l],cc,var,offset=p*64,weighted=False)
        self.active_pk=self.router.packed_pk[:12*(p+72)*4*128].view(12,p+72,4,128)
        torch.cat((self.pk[l][:,:p],ck),1,out=self.active_pk)
        self.active_codes=self.router.code[:12*(p+72)*8].view(12,p+72,8)
        torch.cat((self.codes[l][:,:p],cc),1,out=self.active_codes)
        vv=self.router.vv[:12*(p+72)].view(12,p+72);torch.cat((self.vvar[l][:,:p],var),1,out=vv)
        lut,_=self.router(q,self.active_pk,self.active_codes,vv,self.sw[l],bool(self.strength))
        self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=lut.shape[-1],exact_sparsity=1-lut.shape[-1]/(p+72))
        return lut
    def finish(self):return super().finish()|dict(route_implementation='native summaries/mass/topk; fused feature-score/swap; captured QK/variance; probe gather; reused packed-K calculation workspace')
