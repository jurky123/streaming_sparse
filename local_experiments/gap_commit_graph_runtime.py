"""Capture immutable-history preparation, reading existing current KV tail.
Graph outputs are transient workspaces, consumed before another layer reuses the
shared graph pool. No raw-KV input cache or additional summary cache is added.
"""
import time,torch
from .gap_feature_runtime import GapFeatureRuntime
from .gap_runtime import GapRuntime
from .response_hard import balanced
from .cache_grouping import reorder
from .high_sparse_learn_kernels import pool
class GapCommitGraphRuntime(GapFeatureRuntime):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.commit_graphs={};self.commit_outputs={};self.commit_pool=torch.cuda.graph_pool_handle();self.commit_capture_seconds=0.
    @torch.no_grad()
    def commit(self,l,k,v,cache):
        if self.training_event:return GapRuntime.commit(self,l,k,v,cache)
        # Forward already writes the current KV at n[l]. Once full, no copy.
        if self.n[l]!=self.cap:
            cache['k'][:,self.cap*64:self.cap*64+4608].copy_(k)
            cache['v'][:,self.cap*64:self.cap*64+4608].copy_(v)
        if l not in self.commit_graphs:
            m=self.model[l]
            kin=cache['k'][0,self.cap*64:self.cap*64+4608]
            vin=cache['v'][0,self.cap*64:self.cap*64+4608]
            def prepare():
                kf=kin.transpose(0,1).float();vf=vin.transpose(0,1)
                x=m.group_features(kf[:,None],vf.float()[:,None]).squeeze(1)
                order=balanced(x,[3,3,2,2,2]);xp=reorder(x,order)
                inner=balanced(xp.reshape(-1,64,32),[2,2]).reshape(12,72,64)
                order=order.reshape(12,72,64).gather(2,inner)[:,self.page_order].flatten(1)
                kr=reorder(kf,order);vr=reorder(vf,order)
                kg=kr.reshape(12,72,64,128);vg=vr.reshape_as(kg)
                pk,base,detail=m.pool(kg,vg);cov=m.covariance(kg,vg)
                return kr.transpose(0,1).to(torch.bfloat16),vr.transpose(0,1),pk,base,detail,cov
            start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):warm=prepare()
            torch.cuda.current_stream().wait_stream(stream);del warm
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream,pool=self.commit_pool):out=prepare()
            self.commit_graphs[l]=graph;self.commit_outputs[l]=out;self.commit_capture_seconds+=time.perf_counter()-start
        self.commit_graphs[l].replay();kr,vr,pk,base,detail,cov=self.commit_outputs[l]
        slots=(torch.arange(72,device=k.device)+self.write[l])%self.cap;ids=(slots[:,None]*64+self.offsets).flatten()
        cache['k'][0].index_copy_(0,ids,kr);cache['v'][0].index_copy_(0,ids,vr)
        self.cov[l].index_copy_(1,slots,cov);self.pk[l].index_copy_(1,slots,pk)
        self.codes[l].index_copy_(1,slots,base);self.details[l].index_copy_(1,slots,detail)
        pool(cache,self.basis[l],self.centers[l],self.uk[l],self.uv[l],self.codes[l],self.vvar[l],start=self.write[l],ring=True,weighted=False)
        self.write[l]=(self.write[l]+72)%self.cap;self.n[l]=min(self.cap,self.n[l]+72);self.commits[l]+=1
    def finish(self):return super().finish()|dict(history_prepare='native arithmetic CUDA graph, existing KV tail input, shared transient pool',commit_graph_count=len(self.commit_graphs),commit_capture_seconds=self.commit_capture_seconds)
