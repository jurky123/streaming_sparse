"""Experimental all-page budgets; current summaries are transient per denoise step."""
import math
import torch
import triton as tr
import triton.language as tl
from .compensation_runtime import CompensationRuntime
from .compensation_stable_kernels import _stable_tail
from .compensation_kernels import _merge
from .rollout_kernels import route_scores
MODEL='artifacts/compensation_probe_20261030/model.pt'

@tr.jit
def _budget_mask(LUT,OUT,T:tl.constexpr,K:tl.constexpr,BK:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1);i=tl.arange(0,BK);w=tl.arange(0,16)
    ids=tl.load(LUT+(h*T+t)*K+i,i<K,-999)
    bits=tl.where((i[None,:]<K)&(ids[None,:]//32==w[:,None]),1 << (ids[None,:]%32),0).to(tl.uint32)
    tl.store(OUT+(h*T+t)*16+w,tl.sum(bits,1))

def compensate(q,pk,code,lut,out,lse,basis,center,gamma):
    _,n,h,d=q.shape;p=pk.shape[1];k=lut.shape[-1]
    bits=torch.empty(h,n//64,16,device=q.device,dtype=torch.uint32)
    _budget_mask[(n//64,h)](lut,bits,n//64,k,tr.next_power_of_2(k),num_warps=4)
    tail=torch.empty(h,n,8,device=q.device);lt=torch.empty(h,n,device=q.device)
    _stable_tail[(tr.cdiv(n,64),h)](q,pk,code,bits,tail,lt,p,p,n,h,d,64,16,False,num_warps=4,num_stages=1)
    _merge[(tr.cdiv(n,16),h)](out,lse,tail,lt,basis,center,gamma,out,n,h,d,16,num_warps=4)
    return out,tail,lt

class Kernel:
    def __init__(self,rt,base):self.rt=rt;self.base=base
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            r=self.rt;l=r.active_layer;a=list(args);a[4]=a[5].shape[-1]
            self.base[grid](*a,**kwargs)
            if r.use_comp:
                compensate(a[0],r.active_pk,r.active_codes,a[5],a[7],a[6],r.basis[l],r.centers[l],r.gamma[l])
            r.active_pk=None;r.active_codes=None
        return call

class HighSparseRuntime(CompensationRuntime):
    def __init__(self,engine,policy='global',sparsity=.9,comp=True):
        assert policy in ['global','history10','legacy']
        assert .5<=sparsity<1
        super().__init__(engine,MODEL,mode='raw' if comp else 'zero')
        self.policy=policy;self.target_sparsity=sparsity;self.use_comp=comp
        self.method=f'{policy}_{sparsity:g}_'+('comp' if comp else 'drop')
        self.kernel=Kernel(self,self.kernel.base)
        self.active_pk=None;self.active_codes=None;self.budget_trace={}
    def reset(self):
        super().reset();self.active_pk=None;self.active_codes=None;self.budget_trace={}
    def route(self,l,q,p):
        cache=self.engine.cache[l]
        if self.policy=='legacy':
            z=route_scores(q,self.pk[l][:,:p],self.alpha[l])
            ids=z.topk(72,-1,sorted=False).indices
            current=torch.arange(p,p+72,device=q.device).expand(12,72,72)
            lut=torch.cat((ids,current),-1).sort(-1).values[None].int()
            # Appending current summaries is unnecessary when none are omitted.
        elif self.policy=='history10':
            z=route_scores(q,self.pk[l][:,:p],self.alpha[l])
            ids=z.topk(10,-1,sorted=False).indices
            current=torch.arange(p,p+72,device=q.device).expand(12,72,72)
            lut=torch.cat((ids,current),-1).sort(-1).values[None].int()
        else:lut=None
        # No current-token reordering. Four fixed contiguous 16-token subgroups/page.
        ck=cache['k'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,4,16,128).mean(3)
        self.active_pk=torch.cat((self.pk[l][:,:p],ck),1).contiguous()
        if lut is None:
            z=route_scores(q,self.active_pk,self.alpha[l])
            keep=max(1,int(math.floor((p+72)*(1-self.target_sparsity)+1e-6)))
            own=(torch.arange(72,device=q.device)+p)[None,:,None].expand(12,72,1)
            z.scatter_(-1,own,float('inf'))
            lut=z.topk(keep,-1,sorted=False).indices.sort(-1).values[None].int()
        if self.use_comp:
            vm=cache['v'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,64,128).mean(2)
            cc=((vm-self.centers[l][:,None])@self.basis[l].float()).to(torch.bfloat16)
            self.active_codes=torch.cat((self.codes[l][:,:p],cc),1).contiguous()
        self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=lut.shape[-1],exact_sparsity=1-lut.shape[-1]/(p+72))
        return lut
    def finish(self):
        return super().finish()|dict(selected_history_pages=None,policy=self.policy,target_sparsity=self.target_sparsity,use_compensation=self.use_comp,budget_trace=self.budget_trace,current_summary='4 mean K per chronological page, mean V projected to frozen history-trained rank8 basis; transient recomputed per noisy forward',scope='Sparse calls only; layer0/clean/early dense unchanged. No new training.')
