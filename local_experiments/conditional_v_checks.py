import math,json
import torch
from .conditional_v_kernels import compensate,reconstruct
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.manual_seed(334);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    for p,keep in [(7,2),(19,3)]:
        h,n,d=12,64,128;q=torch.randn(1,n,h,d,device='cuda',dtype=torch.bfloat16);pk=torch.randn(h,p,4,d,device='cuda')*.3
        base=torch.randn(h,p,8,device='cuda',dtype=torch.bfloat16);basis=torch.randn(h,d,8,device='cuda',dtype=torch.bfloat16)*.1;center=torch.randn(h,d,device='cuda')*.1
        lut=torch.arange(keep,device='cuda',dtype=torch.int32)[None,None,None].expand(1,h,1,-1).contiguous();exact=torch.randn_like(q);lse=torch.randn(1,h,n,device='cuda')+5;gamma=torch.ones(h,device='cuda');scale=torch.linspace(-.2,.2,h,device='cuda')
        for mode in ['paired4','haar1']:
            detail=torch.randn((h,p,3,8) if mode=='paired4' else (h,p,8),device='cuda',dtype=torch.bfloat16)*.1;choice=torch.randint(0,3,(h,p),device='cuda',dtype=torch.uint8)
            out,tail,lt=compensate(q,pk,base,detail,choice,lut,exact.clone(),lse,basis,center,gamma,scale)
            codes=reconstruct(base,detail,choice);qq=q[0].transpose(0,1).float();logits=(qq@pk.flatten(1,2).transpose(1,2)/math.sqrt(d)).reshape(h,n,p,4)+math.log(16);logits[:,:,:keep]=-float('inf')
            prob=logits.flatten(-2).softmax(-1);ct=prob@codes.flatten(1,2);logz=logits.flatten(-2).logsumexp(-1)/math.log(2)+scale[:,None]/math.log(2)
            y=ct@basis.float().transpose(1,2)+center[:,None];x=exact[0].transpose(0,1).float();rho=((logz-lse[0])*math.log(2)).sigmoid();ref=(x+rho[...,None]*(y-x)).transpose(0,1)[None]
            error=float((out.float()-ref).square().sum()/ref.square().sum());torch.testing.assert_close(tail,ct,atol=2e-5,rtol=2e-4);torch.testing.assert_close(lt,logz,atol=2e-5,rtol=2e-5);assert error<1e-5
            rows.append(dict(p=p,mode=mode,nmse=error))
    atomic('artifacts/sparse_expansion_20261031/conditional_checks.json',dict(passed=True,rows=rows));print(json.dumps(rows))
if __name__=='__main__':main()
