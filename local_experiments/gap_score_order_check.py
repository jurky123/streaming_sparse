import torch,triton as tr,triton.language as tl,json
from .learned_cache.io import atomic
@tr.jit
def _dot(X,W,O,P:tl.constexpr,N:tl.constexpr,B:tl.constexpr,MODE:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);h=i//(72*P);f=tl.arange(0,16)
    x=tl.load(X+i[:,None]*14+f[None,:],(i[:,None]<N)&(f[None,:]<14),0)
    w=tl.load(W+h[:,None]*14+f[None,:],(i[:,None]<N)&(f[None,:]<14),0)
    if MODE==0:out=tl.sum(x*w,1)
    elif MODE==1:
        out=tl.full((B,),0.,tl.float32)
        for j in tl.static_range(14):
            a=tl.sum(tl.where(f[None,:]==j,x,0.),1);b=tl.sum(tl.where(f[None,:]==j,w,0.),1);out=tl.fma(a,b,out)
    elif MODE==2 or MODE==3:
        acc=tl.full((B,4),0.,tl.float32);g=tl.arange(0,4)
        for j in tl.static_range(4):
            k=g+4*j;a=tl.load(X+i[:,None]*14+k[None,:],(i[:,None]<N)&(k[None,:]<14),0);b=tl.load(W+h[:,None]*14+k[None,:],(i[:,None]<N)&(k[None,:]<14),0)
            acc=tl.fma(a,b,acc)
        a,b=tl.split(tl.reshape(acc,(B,2,2)));a0,a2=tl.split(a);a1,a3=tl.split(b)
        if MODE==2:out=((a0+a1)+a2)+a3
        else:out=(a0+a2)+(a1+a3)
    else:
        v=x*w
        v=tl.sum(tl.reshape(v,(B,8,2)),2);v=tl.sum(tl.reshape(v,(B,4,2)),2);v=tl.sum(tl.reshape(v,(B,2,2)),2);out=tl.sum(v,1)
    tl.store(O+i,out,i<N)
def main():
    torch.set_num_threads(4);torch.manual_seed(75);p=410;x=torch.randn(12,72,p,14,device='cuda');w=torch.randn(12,14,device='cuda');ref=torch.einsum('htpf,hf->htp',x,w);o=torch.empty_like(ref);rows=[]
    for mode in range(5):
        _dot[(tr.cdiv(o.numel(),128),)](x,w,o,p,o.numel(),128,mode,num_warps=4,enable_fp_fusion=False)
        row=dict(mode=mode,exact=torch.equal(o,ref),equal_fraction=float((o==ref).float().mean()),max_error=float((o-ref).abs().max()));rows.append(row);print(json.dumps(row),flush=True)
    atomic('artifacts/gap_route_memory_20261031/score_order.json',dict(rows=rows))
if __name__=='__main__':main()
