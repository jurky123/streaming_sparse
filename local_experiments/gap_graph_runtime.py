"""Exact-arithmetic inference CUDA graphs for conditional replacement routing."""
import math,time,torch
from .gap_runtime import GapRuntime
from .high_sparse_learn import correct
from .high_sparse_gain import score
from .conditional_v_kernels import HAAR
def route_features(prob,kvar,vvar):
    h,t,m,p=prob.shape;mass=prob.mean(2);lm=mass.clamp_min(1e-12).log()
    lp=prob.clamp_min(1e-12).log()
    peak=(prob.max(2).values.clamp_min(1e-12).log()-lm).clamp(0,10)/5
    spread=lp.std(2,unbiased=False).clamp(0,5)/5
    current=(torch.arange(p,device=prob.device)>=p-72).float()[None,None].expand(h,t,-1)
    vv=vvar.float().clamp_min(0).log1p()[:,None].expand(-1,t,-1)
    kv=kvar[:,None].expand(-1,t,-1);lc=lm.clamp(-20,0)/5
    return torch.stack((torch.ones_like(lm),lc,peak,spread,vv,kv,current,current*lc),-1)

def postprocess(logits,qp,base,detail,cov,vvar,kvar,w,gate,haar,training=False):
    h,t,n,p,_=logits.shape;keep=int(math.floor(p*.1+1e-6))
    prob=(logits.logsumexp(-1)+math.log(16)).softmax(-1);mass=prob.mean(2);feat=route_features(prob,kvar,vvar)
    within=logits.softmax(-1);weight=(prob[...,None]*within).mean(2)/mass[...,None].clamp_min(1e-12)
    coefficients=torch.einsum('htpr,ar->htpa',weight,haar)
    conditional=base.float()[:,None]+torch.einsum('htpa,hpad->htpd',coefficients,detail.float())
    # First-order diagonal projected KV-covariance proxy, not an exact page oracle.
    qbar=torch.einsum('htmp,htmd->htpd',prob,qp)/(n*mass[...,None].clamp_min(1e-12))
    residual=qbar*cov.float()[:,None]
    own=(torch.arange(t,device=logits.device)+p-t)[None].expand(h,-1)
    z=mass.detach().clone();z.scatter_(-1,own[...,None],float('inf'));ids=z.topk(keep,-1).indices
    mask=torch.zeros_like(mass,dtype=torch.bool).scatter_(-1,ids,True)
    weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12)
    selected=(weighted[...,None]*conditional).sum(2)/den;allvalue=(mass[...,None]*conditional).sum(2)
    missing=allvalue-selected;diff=conditional-selected[:,:,None];energy=selected.square().sum(-1,keepdim=True).clamp_min(.01)
    align=(diff*missing[:,:,None]).sum(-1)/energy;mag=diff.square().sum(-1)/energy
    extra=torch.stack((mask.float(),(mass/den).clamp(0,2),align.clamp(-10,10)/5,mag.clamp(0,10)/5,(mass*align).clamp(-2,2),(mass*mag).clamp(0,2)),-1)
    s=score(mass,torch.cat((feat,extra),-1),w)
    contribution=mass[...,None]*residual;error=(contribution*(~mask)[...,None]).sum(2)
    # Add reduces predicted error by 2<E,e>-||e||²; dropping costs 2<E,e>+||e||².
    residual_energy=error.square().sum(-1,keepdim=True).clamp_min(1e-8)
    gain=(2*(error[:,:,None]*contribution).sum(-1)+(2*mask.float()-1)*contribution.square().sum(-1))/residual_energy
    s=s+gate.tanh()[:,None,None]*gain.clamp(-5,5)
    with torch.no_grad():
        new,_=correct(mask,s,own)
        def lut(x):return x.int().topk(keep,-1).indices.sort(-1).values[None].int()
        chosen=lut(new)
        if not training:return chosen,None,None,None
        rem=s.masked_fill(~mask,float('inf')).clone();rem.scatter_(-1,own[...,None],float('inf'));drop=rem.argmin(-1);add=s.masked_fill(mask,-float('inf')).argmax(-1)
        alt=mask.clone().scatter_(-1,drop[...,None],False).scatter_(-1,add[...,None],True);baseids=lut(mask);altids=lut(alt)
    margin=s.gather(-1,add[...,None]).squeeze(-1)-s.gather(-1,drop[...,None]).squeeze(-1)
    return chosen,baseids,altids,margin

class GapGraphRuntime(GapRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);h,t,m,p=12,72,self.probe_count,410
        self.graw=torch.empty(h*t*m*p*4,device='cuda');self.gqp=torch.empty(h,t,m,8,device='cuda')
        self.gbase=torch.empty(h*p*8,device='cuda',dtype=torch.bfloat16);self.gdetail=torch.empty(h*p*24,device='cuda',dtype=torch.bfloat16);self.gcov=torch.empty_like(self.gbase)
        self.gvar=torch.empty(h*p,device='cuda');self.gkvar=torch.empty_like(self.gvar);self.gw=torch.empty(h,14,device='cuda');self.ggate=torch.empty(h,device='cuda');self.ghaar=torch.tensor(HAAR,device='cuda',dtype=torch.float32)
        self.aux_bytes=sum(x.numel()*x.element_size() for x in [self.gbase,self.gdetail,self.gcov,self.gvar,self.gkvar,self.gw,self.ggate,self.ghaar])
        self.persistent_bytes+=self.aux_bytes;assert self.persistent_bytes<=self.budget
        self.graphs_new={};self.outputs_new={};self.graph_pool=torch.cuda.graph_pool_handle();self.capture_seconds=0.
    @torch.no_grad()
    def decision(self,l,q,pk,base,detail,cov,vvar):
        if self.training_event:return super().decision(l,q,pk,base,detail,cov,vvar)
        m=self.model[l];h,p,_,d=pk.shape;t=q.shape[1]//64;n=self.probe_count
        query=q[0].transpose(0,1).float().reshape(h,t,64,d).index_select(2,self.probe_offsets)
        logits=(query.reshape(h,t*n,d)@pk.flatten(1,2).transpose(1,2)/math.sqrt(d)).reshape(h,t,n,p,4)
        qp=query@m.basis.to(torch.bfloat16).float()[:,None]/math.sqrt(d);kvar=(pk-pk.mean(2,keepdim=True)).square().mean((2,3)).log1p()
        args=(self.graw[:logits.numel()].view_as(logits),self.gqp,self.gbase[:base.numel()].view_as(base),self.gdetail[:detail.numel()].view_as(detail),self.gcov[:cov.numel()].view_as(cov),self.gvar[:vvar.numel()].view_as(vvar),self.gkvar[:kvar.numel()].view_as(kvar),self.gw,self.ggate,self.ghaar)
        for dest,src in zip(args[:-1],(logits,qp,base,detail,cov,vvar,kvar,m.route,m.residual_gate)):dest.copy_(src)
        if p not in self.graphs_new:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=postprocess(*args)
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.graph_pool):out=postprocess(*args)
            self.graphs_new[p]=graph;self.outputs_new[p]=out;self.capture_seconds+=time.perf_counter()-start
        self.graphs_new[p].replay();return self.outputs_new[p]
    def finish(self):return super().finish()|dict(route_implementation='conditional replacement route CUDA graphs, native arithmetic',graph_extra_kv_aux_bytes=self.aux_bytes,graph_capture_seconds=self.capture_seconds,graph_count=len(self.graphs_new))
