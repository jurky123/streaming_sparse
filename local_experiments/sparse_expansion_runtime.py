"""Bounded probe-count and candidate-pool refinement experiments; backbone frozen."""
import math,time
import torch,triton as tr,triton.language as tl
from .route_vector_runtime import VectorRuntime
from .route_whole_runtime import WholeRouter,_load_current,_load_history
from .high_sparse_gain_runtime import GainRuntime
from .high_sparse_learn import features,swap_features,correct
from .high_sparse_gain import score
from .high_sparse_learn_kernels import pool
from .route_norm_kernels import _logz

@tr.jit
def _gather_many(Q,I,OUT,M:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B);d=x%128;m=x//128%M;t=x//(128*M)%72;h=x//(128*M*72)
    offset=tl.load(I+m);src=((t*64+offset)*12+h)*128+d
    tl.store(OUT+x,tl.load(Q+src,x<12*72*M*128,0).to(tl.float32),x<12*72*M*128)

def refine(mass,feat,code,w,own,keep,steps,pool_size=8):
    z=mass.clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    initial=mask.clone();sf=swap_features(feat,mass,code.float(),mask);s=score(mass,sf,w)
    rem=s.masked_fill(~mask,float('inf'));rem.scatter_(-1,own[...,None],float('inf'))
    low,drop=rem.min(-1);high,add=s.masked_fill(mask,-float('inf')).max(-1);change=high>low
    if steps==0:return mask,torch.zeros_like(low)
    mask=mask.clone().scatter_(-1,drop[...,None],~change[...,None]).scatter_(-1,add[...,None],change[...,None]);changes=change.float()
    if steps==1:return mask,changes
    # Later iterations only examine 8 original additions + 8 original removals.
    # No reversal or revisiting: at most `steps` distinct pairs, same page budget.
    adds=s.masked_fill(initial,-float('inf')).topk(pool_size,-1).indices
    drops=rem.topk(pool_size,-1,largest=False).indices
    allowed_a=(adds!=add[...,None])|(~change[...,None]);allowed_d=(drops!=drop[...,None])|(~change[...,None])
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12);num=weighted@code.float();allv=mass@code.float()
    cand=torch.cat((adds,drops),-1);cm=mass.gather(-1,cand);cf=feat.gather(2,cand[...,None].expand(-1,-1,-1,8))
    cc=code.float()[:,None].expand(-1,mass.shape[1],-1,-1).gather(2,cand[...,None].expand(-1,-1,-1,8))
    membership=torch.cat((torch.zeros_like(adds),torch.ones_like(drops)),-1).float()
    for _ in range(1,steps):
        selected=num/den;missing=allv-selected;delta=cc-selected[:,:,None];energy=selected.square().sum(-1,keepdim=True).clamp_min(.01)
        align=(delta*missing[:,:,None]).sum(-1)/energy;mag=delta.square().sum(-1)/energy
        extra=torch.stack((membership,(cm/den).clamp(0,2),align.clamp(-10,10)/5,mag.clamp(0,10)/5,(cm*align).clamp(-2,2),(cm*mag).clamp(0,2)),-1)
        cs=cm.clamp_min(1e-12).log()+torch.tanh(torch.einsum('htcf,hf->htc',torch.cat((cf,extra),-1),w))
        sa=cs[...,:pool_size].masked_fill(~allowed_a,-float('inf'));sd=cs[...,pool_size:].masked_fill(~allowed_d,float('inf'))
        hi=sa.max(-1).values;lo=sd.min(-1).values;active=(hi>lo)&torch.isfinite(hi)&torch.isfinite(lo)
        ai=adds.masked_fill(sa!=hi[...,None],2147483647).min(-1).values;di=drops.masked_fill(sd!=lo[...,None],2147483647).min(-1).values
        ai=torch.where(active,ai,own);di=torch.where(active,di,own)
        oldd=mask.gather(-1,di[...,None]);mask=mask.scatter(-1,di[...,None],oldd&~active[...,None]).scatter(-1,ai[...,None],True)
        ma=mass.gather(-1,ai[...,None]);md=mass.gather(-1,di[...,None]);ca=code.float()[:,None].expand(-1,mass.shape[1],-1,-1).gather(2,ai[...,None,None].expand(-1,-1,1,8)).squeeze(2);cd=code.float()[:,None].expand(-1,mass.shape[1],-1,-1).gather(2,di[...,None,None].expand(-1,-1,1,8)).squeeze(2)
        num=num+active[...,None]*(ma*ca-md*cd);den=(den+active[...,None]*(ma-md)).clamp_min(1e-12)
        allowed_a &= (adds!=ai[...,None])|(~active[...,None]);allowed_d &= (drops!=di[...,None])|(~active[...,None]);changes+=active.float()
    return mask,changes

class ExpandedRouter(WholeRouter):
    def __init__(self,probes,steps):
        super().__init__();self.probe_count=probes;self.steps=steps
        old=self.probes.numel()*self.probes.element_size()
        self.probes=torch.linspace(0,63,probes,device='cuda').round().long();self.kv_aux_bytes+=self.probes.numel()*8-old
        self.raw=torch.empty(12*72*probes*410*4,device='cuda');self.query=torch.empty(12,72*probes,128,device='cuda')
    @torch.no_grad()
    def route(self,l,q,p):
        self.layer.fill_(l);m=self.probe_count;_gather_many[(tr.cdiv(self.query.numel(),1024),)](q,self.probes,self.query,m,1024)
        total=p+72;key=(total,True);pk=self.packed_pk[:12*total*512].view(12,total,4,128);code=self.code[:12*total*8].view(12,total,8);vv=self.vv[:12*total].view(12,total)
        def pipeline():
            _load_current[(tr.cdiv(self.k.numel(),1024),)](self.ptrs,self.layer,self.k,self.v,self.basis,self.center,self.weight,p,1024)
            _load_history[(p,12)](self.ptrs,self.layer,pk,code,vv,p,num_warps=4)
            ck=self.k[0].transpose(0,1).reshape(12,72,4,16,128).mean(3);vm=self.v[0].transpose(0,1).reshape(12,72,64,128).mean(2)
            cc=((vm-self.center[:,None])@self.basis.float()).to(torch.bfloat16)
            pk[:,p:].copy_(ck);code[:,p:].copy_(cc);vv[:,p:].copy_(self.current_var)
            raw=self.raw[:12*72*m*total*4].view(12,72*m,total*4);torch.bmm(self.query,pk.flatten(1,2).transpose(1,2),out=raw)
            z=torch.empty(12,72,m,total,device='cuda');_logz[(tr.cdiv(z.numel(),256),)](raw,z,z.numel(),0,0,enable_fp_fusion=False)
            prob=z.softmax(-1);mass=prob.mean(2);feat=features(prob,pk,vv);own=(torch.arange(72,device='cuda')+p)[None].expand(12,-1);keep=int(math.floor(total*.1+1e-6))
            mask,changes=refine(mass,feat,code,self.weight,own,keep,self.steps)
            lut=mask.int().topk(keep,-1,sorted=False).indices.sort(-1).values[None].int()
            return lut,changes
        if key not in self.graphs:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=pipeline()
            torch.cuda.current_stream().wait_stream(stream);del warm
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=pipeline()
            self.graphs[key]=g;self.outputs[key]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs[key].replay();return self.outputs[key][0],pk,code

class ExpansionRuntime(VectorRuntime):
    def __init__(self,engine,checkpoint,probes=4,steps=1):
        assert probes in [4,8,16,64] and steps in [0,1,2,4]
        GainRuntime.__init__(self,engine,checkpoint,1.);self.probe_count=probes;self.refine_steps=steps
        self.router=ExpandedRouter(probes,steps);self.persistent_bytes+=self.router.kv_aux_bytes
        assert self.persistent_bytes<=self.budget;self.router.bind(self);self.route_meta=None
    def finish(self):return super().finish()|dict(route_implementation='configurable probe graph, native features; fixed 8+8 shortlist after first swap',probes=self.probe_count,correction_steps=self.refine_steps)
