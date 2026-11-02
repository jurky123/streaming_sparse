"""General-probe graph route; retain native arithmetic and small KV input budget."""
import math,time,torch
from .all_sparse_inference import InferenceRuntime
from .high_sparse_learn import swap_features
from .high_sparse_gain import score
def postprocess(raw,code,vvar,kvar,w,p,keep,enabled,m):
    logits=(raw/math.sqrt(128)).reshape(12,72,m,p,4)
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


class GraphInferenceRuntime(InferenceRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.graph_inputs={};self.graphs2={};self.graph_outputs={};self.capture_seconds2=0.
        self.graph_code=torch.empty(12*410*8,device='cuda',dtype=torch.bfloat16)
        self.graph_var=torch.empty(12*410,device='cuda');self.graph_kvar=torch.empty_like(self.graph_var);self.graph_weight=torch.empty(12,14,device='cuda')
        self.extra_aux=sum(x.numel()*x.element_size() for x in [self.graph_code,self.graph_var,self.graph_kvar,self.graph_weight]);self.persistent_bytes+=self.extra_aux
        assert self.persistent_bytes<=self.budget
    @torch.no_grad()
    def decision(self,l,q,pk,base,vvar):
        if self.training_event:return super().decision(l,q,pk,base,vvar)
        p=pk.shape[1];m=self.probe_count;keep=int(math.floor(p*.1+1e-6))
        query=q[0].transpose(0,1).float().reshape(12,72,64,128).index_select(2,self.probe_offsets)
        raw=query.reshape(12,72*m,128)@pk.flatten(1,2).transpose(1,2)
        kvar=(pk-pk.mean(2,keepdim=True)).square().mean((2,3)).log1p()
        if p not in self.graph_inputs:self.graph_inputs[p]=torch.empty_like(raw)
        rawbuf=self.graph_inputs[p];rawbuf.copy_(raw)
        code=self.graph_code[:base.numel()].view_as(base);vv=self.graph_var[:vvar.numel()].view_as(vvar);kv=self.graph_kvar[:kvar.numel()].view_as(kvar)
        code.copy_(base);vv.copy_(vvar);kv.copy_(kvar);self.graph_weight.copy_(self.model[l].route)
        args=(rawbuf,code,vv,kv,self.graph_weight,p,keep,True,m)
        if p not in self.graphs2:
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=postprocess(*args)
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):out=postprocess(*args)
            self.graphs2[p]=graph;self.graph_outputs[p]=out;self.capture_seconds2+=time.perf_counter()-start
        self.graphs2[p].replay();return self.graph_outputs[p][0],None,None,None
    def finish(self):return super().finish()|dict(route_implementation='native generalized probe arithmetic in shared CUDA graphs; zero-copy historical KV; no training branch outputs',graph_extra_aux_bytes=self.extra_aux,graph_capture_seconds=self.capture_seconds2,graph_shapes=list(self.graphs2),graph_workspace='query logits and intermediate routing features are computation workspace; code/variance/weights included in persistent budget')
