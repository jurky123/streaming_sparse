"""Native arithmetic features and native score GEMV, with fused memory traffic."""
import math
import torch,triton as tr
from .route_norm_kernels import logz
from .route_exact_features import _features_exact
from .route_fused_kernels import _lut
from .route_whole_runtime import WholeRouter,WholeRuntime
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_learn_kernels import pool
from .route_fused_kernels import gather
from .route_whole_runtime import _load_current,_load_history
import time

def exact_postprocess(raw,code,vvar,kvar,w,p,keep,enabled):
    prob=logz(raw,p).softmax(-1);mass=prob.mean(2)
    own=(torch.arange(72,device=raw.device)+p-72)[None].expand(12,-1)
    z=mass.clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    if not enabled:return ids.sort(-1).values[None].int(),None
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=weighted@code.float()/den;allvalue=mass@code.float()
    spread=prob.clamp_min(1e-12).log().std(2,unbiased=False).clamp(0,5)/5
    sf=torch.empty(12,72,p,14,device=raw.device)
    _features_exact[(864,)](prob,mass,code,vvar,kvar,w,mask,den,selected,allvalue,(mass/den).clamp(0,2),spread,sf,p,tr.next_power_of_2(p),num_warps=4,enable_fp_fusion=False)
    s=mass.clamp_min(1e-12).log()+torch.tanh(torch.einsum('htpf,hf->htp',sf,w))
    out=torch.empty((1,12,72,keep),device=raw.device,dtype=torch.int32)
    _lut[(864,)](s,mask,ids,out,p,keep,tr.next_power_of_2(p),tr.next_power_of_2(keep),num_warps=4)
    return out,(mass,sf,mask,own,s)
class ExactRouter(WholeRouter):
    postprocess=staticmethod(exact_postprocess)
    @torch.no_grad()
    def route(self,l,q,p):
        self.layer.fill_(l);gather(q,self.query);total=p+72;key=(total,True)
        pk=self.packed_pk[:12*total*512].view(12,total,4,128)
        code=self.code[:12*total*8].view(12,total,8);vv=self.vv[:12*total].view(12,total)
        def pipeline():
            _load_current[(tr.cdiv(self.k.numel(),1024),)](self.ptrs,self.layer,self.k,self.v,self.basis,self.center,self.weight,p,1024)
            _load_history[(p,12)](self.ptrs,self.layer,pk,code,vv,p,num_warps=4)
            ck=self.k[0].transpose(0,1).reshape(12,72,4,16,128).mean(3)
            vm=self.v[0].transpose(0,1).reshape(12,72,64,128).mean(2)
            cc=((vm-self.center[:,None])@self.basis.float()).to(torch.bfloat16)
            pk[:,p:].copy_(ck);code[:,p:].copy_(cc);vv[:,p:].copy_(self.current_var)
            raw=self.raw[:12*288*total*4].view(12,288,total*4)
            torch.bmm(self.query,pk.flatten(1,2).transpose(1,2),out=raw)
            kv=self.kvar[:12*total].view(12,total)
            torch.log1p((pk-pk.mean(2,keepdim=True)).square().mean((2,3)),out=kv)
            return self.postprocess(raw,code,vv,kv,self.weight,total,int(math.floor(total*.1+1e-6)),True)
        if key not in self.graphs:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=pipeline()
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):out=pipeline()
            self.graphs[key]=graph;self.outputs[key]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs[key].replay();return self.outputs[key][0],pk,code

class ExactRuntime(WholeRuntime):
    def __init__(self,engine,checkpoint,strength=1.):
        assert strength==1.
        GainRuntime.__init__(self,engine,checkpoint,strength)
        if not hasattr(engine,'shared_exact_router'):engine.shared_exact_router=ExactRouter()
        self.router=engine.shared_exact_router;self.persistent_bytes+=self.router.kv_aux_bytes
        assert self.persistent_bytes<=self.budget;self.router.bind(self);self.route_meta=None
    def route(self,l,q,p):
        pool(self.engine.cache[l],self.basis[l],self.centers[l],self.uk[l],self.uv[l],self.router.code,self.router.current_var,offset=p*64,weighted=False)
        return super().route(l,q,p)
    def finish(self):return super().finish()|dict(route_implementation='whole-route graph; native-order logz/feature reductions/divisions; native std and score GEMV; fused feature traffic and swap LUT')
