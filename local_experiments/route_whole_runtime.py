"""One shape-shared whole-route graph; pointer aliases, no extra historical KV copies."""
import math,time
import torch,triton as tr,triton.language as tl
from .route_norm_runtime import NormRouter,NormRuntime
from .route_fused_kernels import gather
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_learn_kernels import pool
@tr.jit
def _load_current(PTR,LAYER,K,V,BASIS,CENTER,WEIGHT,P:tl.constexpr,B:tl.constexpr):
    l=tl.load(LAYER);i=tl.program_id(0)*B+tl.arange(0,B);n=4608*12*128
    kp=tl.load(PTR+l*8+0).to(tl.pointer_type(tl.bfloat16));vp=tl.load(PTR+l*8+1).to(tl.pointer_type(tl.bfloat16))
    tl.store(K+i,tl.load(kp+P*64*12*128+i,i<n,0).to(tl.float32),i<n)
    tl.store(V+i,tl.load(vp+P*64*12*128+i,i<n,0).to(tl.float32),i<n)
    # Disjoint small inputs; piggyback on the first blocks.
    bp=tl.load(PTR+l*8+5).to(tl.pointer_type(tl.bfloat16));cp=tl.load(PTR+l*8+6).to(tl.pointer_type(tl.float32));wp=tl.load(PTR+l*8+7).to(tl.pointer_type(tl.float32))
    tl.store(BASIS+i,tl.load(bp+i,i<12*128*8,0),i<12*128*8)
    tl.store(CENTER+i,tl.load(cp+i,i<12*128,0),i<12*128)
    tl.store(WEIGHT+i,tl.load(wp+i,i<12*14,0),i<12*14)
@tr.jit
def _load_history(PTR,LAYER,PK,CODE,VV,P:tl.constexpr):
    p=tl.program_id(0);h=tl.program_id(1);x=tl.arange(0,512);r=tl.arange(0,8);l=tl.load(LAYER)
    kp=tl.load(PTR+l*8+2).to(tl.pointer_type(tl.float32));cp=tl.load(PTR+l*8+3).to(tl.pointer_type(tl.bfloat16));vp=tl.load(PTR+l*8+4).to(tl.pointer_type(tl.bfloat16))
    tl.store(PK+(h*(P+72)+p)*512+x,tl.load(kp+(h*338+p)*512+x))
    tl.store(CODE+(h*(P+72)+p)*8+r,tl.load(cp+(h*338+p)*8+r))
    tl.store(VV+h*(P+72)+p,tl.load(vp+h*338+p).to(tl.float32))
class WholeRouter(NormRouter):
    def __init__(self):
        super().__init__()
        self.ptrs=torch.empty(30,8,device='cuda',dtype=torch.int64);self.layer=torch.empty((),device='cuda',dtype=torch.int32)
        self.basis=torch.empty(12,128,8,device='cuda',dtype=torch.bfloat16);self.center=torch.empty(12,128,device='cuda')
        # Same FP32 cast intermediates as native summaries. Computed afresh each call.
        self.k=torch.empty(1,4608,12,128,device='cuda');self.v=torch.empty_like(self.k)
        self.current_var=torch.empty(12,72,device='cuda',dtype=torch.bfloat16)
        self.kv_aux_bytes+=sum(t.numel()*t.element_size() for t in [self.ptrs,self.layer,self.basis,self.center])
        self.bound=None
    def bind(self,rt):
        pointers=[[0]*8]
        for l in range(1,30):
            pointers.append([x.data_ptr() for x in [rt.engine.cache[l]['k'],rt.engine.cache[l]['v'],rt.pk[l],rt.codes[l],rt.vvar[l],rt.basis[l],rt.centers[l],rt.sw[l]]])
        self.ptrs.copy_(torch.tensor(pointers,dtype=torch.int64));self.bound=id(rt)
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
            pool(dict(k=self.k,v=self.v),self.basis,self.center,self.center,self.center,cc,self.current_var,weighted=False)
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
class WholeRuntime(NormRuntime):
    def __init__(self,engine,checkpoint,strength=1.):
        assert strength==1.,'Whole graph currently targets enabled correction only'
        GainRuntime.__init__(self,engine,checkpoint,strength)
        if not hasattr(engine,'shared_whole_router'):engine.shared_whole_router=WholeRouter()
        self.router=engine.shared_whole_router;self.persistent_bytes+=self.router.kv_aux_bytes
        assert self.persistent_bytes<=self.budget,(self.persistent_bytes,self.budget)
        self.router.bind(self);self.route_meta=None
    def route(self,l,q,p):
        assert self.router.bound==id(self)
        lut,self.active_pk,self.active_codes=self.router.route(l,q,p)
        self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=lut.shape[-1],exact_sparsity=1-lut.shape[-1]/(p+72))
        return lut
    def finish(self):return super().finish()|dict(route_implementation='whole-route CUDA graph with dynamic layer pointer aliases; native-order summaries/logz; fused score/LUT',graph_extra_input_bytes=32644)
