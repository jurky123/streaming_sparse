"""Frozen summary geometry, four real query probes per tile; no dense oracle online."""
import math
import torch
from .high_sparse_runtime import HighSparseRuntime
class ProbeHighSparseRuntime(HighSparseRuntime):
    def route(self,l,q,p):
        assert self.policy=='global'
        cache=self.engine.cache[l];device=q.device
        ck=cache['k'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,4,16,128).mean(3)
        self.active_pk=torch.cat((self.pk[l][:,:p],ck),1).contiguous()
        # Four deterministic spread probes; full token queries still compute attention.
        indices=torch.tensor([0,21,42,63],device=device)
        query=q[0].transpose(0,1).float().reshape(12,72,64,128).index_select(2,indices).reshape(12,288,128)
        z=(query@self.active_pk.flatten(1,2).transpose(1,2)/math.sqrt(128)).reshape(12,72,4,p+72,4).logsumexp(-1)
        scores=z.softmax(-1).mean(2) # Common log16 is constant across every page.
        keep=max(1,int(math.floor((p+72)*(1-self.target_sparsity)+1e-6)))
        own=(torch.arange(72,device=device)+p)[None,:,None].expand(12,72,1);scores.scatter_(-1,own,float('inf'))
        lut=scores.topk(keep,-1,sorted=False).indices.sort(-1).values[None].int()
        if self.use_comp:
            vm=cache['v'][0,p*64:p*64+4608].transpose(0,1).float().reshape(12,72,64,128).mean(2)
            cc=((vm-self.centers[l][:,None])@self.basis[l].float()).to(torch.bfloat16)
            self.active_codes=torch.cat((self.codes[l][:,:p],cc),1).contiguous()
        self.budget_trace[p]=dict(history_pages=p,total_pages=p+72,selected_pages=lut.shape[-1],exact_sparsity=1-lut.shape[-1]/(p+72))
        return lut
    def finish(self):return super().finish()|dict(router='4 real query probes/tile; normalize approximate page partition per query then average; no variance correction, no retraining, current and history summaries both included')
