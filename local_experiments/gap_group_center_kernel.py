"""Native-order page/subgroup means and centering, shared by K/V pooling."""
import torch,triton as tr,triton.language as tl
@tr.jit
def _sum4(x,B:tl.constexpr):
    g=tl.arange(0,4)
    a=tl.sum(tl.where(g[:,None]==0,x,0.),0);b=tl.sum(tl.where(g[:,None]==1,x,0.),0)
    c=tl.sum(tl.where(g[:,None]==2,x,0.),0);d=tl.sum(tl.where(g[:,None]==3,x,0.),0)
    return ((a+b)+c)+d
@tr.jit
def _group_center(K,V,KM,VM,KC,KG,VG,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);h=row//P;p=row%P;d=tl.program_id(1)*B+tl.arange(0,B);g=tl.arange(0,4)
    for group in tl.static_range(4):
        ak=tl.full((4,B),0.,tl.float32);av=tl.full((4,B),0.,tl.float32)
        for j in tl.static_range(4):
            t=group*16+g+4*j
            k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]);v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:])
            ak=ak+k;av=av+v
        mk=_sum4(ak,B)/16.;mv=_sum4(av,B)/16.
        for j in tl.static_range(4):
            t=group*16+g+4*j
            ki=h*KH+p*KP+t[:,None]*KT+d[None,:];vi=h*VH+p*VP+t[:,None]*VT+d[None,:]
            goff=((h*4+group)*P+p)*16*128+(g+4*j)[:,None]*128+d[None,:]
            tl.store(KG+goff,tl.load(K+ki)-mk[None,:]);tl.store(VG+goff,tl.load(V+vi)-mv[None,:])
    mk=tl.load(KM+row*128+d)
    for j in tl.static_range(16):
        t=g+4*j;ki=h*KH+p*KP+t[:,None]*KT+d[None,:]
        tl.store(KC+ki,tl.load(K+ki)-mk[None,:])
def group_centers(k,v,block=32):
    h,p,t,d=k.shape;assert (t,d)==(64,128) and k.dtype==v.dtype==torch.float32
    # Preserve native mean64 reduction schedule.
    km=k.mean(2,keepdim=True);vm=v.mean(2,keepdim=True)
    kc=torch.empty_like(k);kg=torch.empty(h,4,p,16,d,device=k.device).permute(0,2,1,3,4);vg=torch.empty_like(kg)
    _group_center[(h*p,tr.cdiv(d,block))](k,v,km,vm,kc,kg,vg,p,*k.stride()[:3],*v.stride()[:3],block,num_warps=4,enable_fp_fusion=False)
    return km,vm,kc,kg,vg
