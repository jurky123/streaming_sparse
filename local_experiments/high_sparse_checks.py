"""Variable LUT stride and masked compensation formula checks, no trajectory claim."""
import json
import torch
from .cache_sparse_prototype import load_kernel
from .high_sparse_runtime import compensate,_budget_mask
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(117);torch.backends.cuda.matmul.allow_tf32=False
    rows=[];h,d,n=12,128,128
    for p in [216,288,360,410]:
        k=torch.randn(1,p*64,h,d,device='cuda',dtype=torch.bfloat16)*.3;v=torch.randn_like(k)
        q=torch.randn(1,n,h,d,device='cuda',dtype=torch.bfloat16)
        pk=k[0].transpose(0,1).float().reshape(h,p,4,16,d).mean(3)
        basis=torch.randn(h,d,8,device='cuda',dtype=torch.bfloat16)*.1;center=torch.randn(h,d,device='cuda')*.1
        vm=v[0].transpose(0,1).float().reshape(h,p,64,d).mean(2)
        code=((vm-center[:,None])@basis.float()).to(torch.bfloat16);gamma=torch.ones(h,device='cuda')
        for fraction in [.2,.15,.1,.075]:
            keep=max(1,int(p*fraction+1e-6));ids=torch.rand(h,n//64,p,device='cuda').topk(keep,-1).indices.sort(-1).values
            lut=ids[None].int().contiguous();out=torch.empty_like(q);lse=torch.empty(1,h,n,device='cuda')
            load_kernel()[(n//64,h)](q,k,v,d**-.5,keep,lut,lse,out,h,n,p*64,n//64,d,64,64,num_warps=4,num_stages=3)
            original=out.clone();errors=[]
            for head in [0,11]:
                qi=65;tokens=(ids[head,qi//64,:,None]*64+torch.arange(64,device='cuda')).flatten()
                logits=q[0,qi,head].float()@k[0,tokens,head].float().T*d**-.5
                ref=logits.softmax(-1)@v[0,tokens,head].float()
                errors.append(float((original[0,qi,head].float()-ref).square().sum()/ref.square().sum()))
            assert max(errors)<1e-4
            result,tail,lt=compensate(q,pk,code,lut,out,lse,basis,center,gamma)
            qq=q[0].transpose(0,1).float();z=(qq@pk.flatten(1,2).transpose(1,2)/d**.5).reshape(h,n,p,4).logsumexp(-1)+__import__('math').log(16)
            mask=torch.zeros(h,n//64,p,device='cuda',dtype=torch.bool).scatter_(-1,ids,True).repeat_interleave(64,1)
            z.masked_fill_(mask,-torch.inf);ta=z.softmax(-1)@code.float();tl=z.logsumexp(-1)/__import__('math').log(2)
            ref=ta@basis.float().transpose(1,2)+center[:,None];rho=(1+torch.exp2(lse[0]-tl)).reciprocal()
            ref=original[0].transpose(0,1).float()+(ref-original[0].transpose(0,1).float())*rho[...,None]
            error=float((result[0].transpose(0,1).float()-ref).square().sum()/ref.square().sum())
            assert error<1e-5 and torch.isfinite(result).all(),(p,keep,error)
            rows.append(dict(pages=p,keep=keep,exact_nmse=max(errors),compensated_nmse=error));print(json.dumps(rows[-1]),flush=True)
    atomic('artifacts/high_sparse_20261030/kernel_checks.json',dict(passed=True,rows=rows))
if __name__=='__main__':main()
