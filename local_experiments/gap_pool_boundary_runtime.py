"""Compile current summary with a physical BF16 basis input boundary."""
import time,torch
from .gap_ordered_runtime import GapFlatOrderedRuntime
def summary(k,v,bank,pool_k,pool_v,pool_feature,basis_bf16,center,haar_input):
    h,p,_,d=k.shape;k=k.float();v=v.float()
    kc=k-k.mean(-2,keepdim=True)
    f=torch.einsum('hptd,hrd->hptr',kc,bank)/128**.5
    f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    f=f.reshape(h,p,4,16,32);kg=k.reshape(h,p,4,16,d);vg=v.reshape_as(kg)
    z=torch.einsum('hprtd,hrd->hprt',kg-kg.mean(-2,keepdim=True),pool_k)
    z=z+torch.einsum('hprtd,hrd->hprt',vg-vg.mean(-2,keepdim=True),pool_v)+torch.einsum('hprtf,hrf->hprt',f,pool_feature)
    z=z-z.mean(-1,keepdim=True);a=(2*torch.tanh(z/2)).softmax(-1)
    pk=(a[...,None]*kg).sum(-2);pv=(a[...,None]*vg).sum(-2)
    # Preserve page-level K mean and encoded V mean. No new cache bytes.
    pk=pk-pk.mean(2,keepdim=True)+k.mean(2,keepdim=True)
    basis=basis_bf16.float()
    codes=torch.einsum('hprd,hdk->hprk',pv-center[:,None,None],basis)
    base=torch.einsum('hpd,hdk->hpk',v.mean(2)-center[:,None],basis).to(torch.bfloat16)
    haar=haar_input
    detail=(torch.einsum('kr,hprd->hpkd',haar,codes)/4).to(torch.bfloat16)
    aa=torch.einsum('hptd,hdr->hptr',k,basis).reshape(h,p,4,16,8)
    bb=torch.einsum('hptd,hdr->hptr',v,basis).reshape_as(aa)
    cov=((aa-aa.mean(3,keepdim=True))*(bb-bb.mean(3,keepdim=True))).mean((2,3)).to(torch.bfloat16).contiguous()
    return pk.contiguous(),base.contiguous(),detail.contiguous(),cov

compiled_summary=torch.compile(summary,fullgraph=True,options={'triton.cudagraphs':False,'emulate_precision_casts':True})
class GapPoolBoundaryRuntime(GapFlatOrderedRuntime):
    @torch.no_grad()
    def current_summary(self,l):
        p=self.n[l];key=(l,p);cache=self.engine.cache[l];m=self.model[l]
        if key not in self.pool_graphs:
            k=cache['k'][0,p*64:p*64+4608].transpose(0,1).reshape(12,72,64,128)
            v=cache['v'][0,p*64:p*64+4608].transpose(0,1).reshape_as(k)
            def run():
                pk,base,detail,cov=compiled_summary(k,v,m.bank,m.pool_k,m.pool_v,m.pool_feature,m.basis.to(torch.bfloat16),m.center,self.ghaar)
                for dest,src in zip(self.current_outputs,(pk,base,detail,cov)):dest.copy_(src)
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):run()
            torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.summary_pool):run()
            self.pool_graphs[key]=graph;self.pool_capture_seconds+=time.perf_counter()-start
        self.pool_graphs[key].replay();return self.current_outputs
    def finish(self):return super().finish()|dict(current_pool='compiled current summary, explicit BF16 basis input boundary')
