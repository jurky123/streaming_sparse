"""Register-bounded latest summary kernels: page features, rep pool, page codes, rep stats."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .fused_summary import _features
from .ops import Summary
@tr.jit
def _origin_dot(x,w):
 hi=w.to(tl.bfloat16);rest=w-hi.to(tl.float32);lo=rest.to(tl.bfloat16);tail=(rest-lo.to(tl.float32)).to(tl.bfloat16)
 acc=tl.dot(x,tail);acc=tl.dot(x,lo,acc);return tl.dot(x,hi,acc)
@tr.jit
def _features_bf16(K,V,BANK,GV,GATE,FEAT,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,CHEAP:tl.constexpr):
 row=tl.program_id(0);h=row//P;p=row%P;t=tl.arange(0,64);d=tl.arange(0,128);f=tl.arange(0,32)
 k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]);bank=tl.load(BANK+h*32*128+f[None,:]*128+d[:,None])
 raw=_origin_dot(k,bank);score=(raw-tl.sum(raw,0)[None,:]/64.)*0.08838834764831843
 rms=tl.maximum(lib.sqrt_rn(tl.sum(tl.sum(score*score,0),0)/2048.),1.e-6);feat=score/rms
 if not CHEAP:
  v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]);gg=tl.load(GV+h*128*32+d[:,None]*32+f[None,:]);vp=_origin_dot(v,gg)
  prob=tl.exp(score-tl.max(score,0)[None,:]);prob=prob/tl.sum(prob,0)[None,:];mean=tl.sum(prob*vp,0);resp=prob*64.*(vp-mean[None,:])
  rrms=tl.maximum(lib.sqrt_rn(tl.sum(tl.sum(resp*resp,0),0)/2048.),1.e-6);gate=.1*lib.tanh(tl.load(GATE+h*32+f));feat=feat+gate[None,:]*(resp/rrms)
 tl.store(FEAT+(row*64+t[:,None])*32+f[None,:],feat)
@tr.jit
def _pool_rep(K,V,F,WK,WV,WF,PK,PV,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr):
 rep=tl.program_id(0);page=rep//4;r=rep%4;h=page//P;p=page%P;t=tl.arange(0,16);d=tl.arange(0,128);f=tl.arange(0,32)
 k=tl.load(K+h*KH+p*KP+(r*16+t[:,None])*KT+d[None,:]).to(tl.float32);v=tl.load(V+h*VH+p*VP+(r*16+t[:,None])*VT+d[None,:]).to(tl.float32)
 feat=tl.load(F+(page*64+r*16+t[:,None])*32+f[None,:]);wk=tl.load(WK+(h*4+r)*128+d);wv=tl.load(WV+(h*4+r)*128+d);wf=tl.load(WF+(h*4+r)*32+f)
 z=tl.sum((k-tl.sum(k,0)[None,:]/16.)*wk[None,:],1)+tl.sum((v-tl.sum(v,0)[None,:]/16.)*wv[None,:],1)
 z=z+tl.sum(feat*wf[None,:],1);z=2.*lib.tanh((z-tl.sum(z,0)/16.)/2.);a=tl.exp(z-tl.max(z,0));a=a/tl.sum(a,0)
 pk=tl.sum(a[:,None]*k,0);pv=tl.sum(a[:,None]*v,0);tl.store(PK+rep*128+d,pk);tl.store(PV+rep*128+d,pv)
@tr.jit
def _page_summary(K,V,RAW,PV,BASIS,PK,MU,DETAIL,ST,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
 row=tl.program_id(0);h=row//P;p=row%P;t=tl.arange(0,64);d=tl.arange(0,128);r=tl.arange(0,4);rank=tl.arange(0,8)
 k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.float32);km=tl.sum(k,0)/64.
 v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.float32);mean=tl.sum(v,0)/64.;mu=mean.to(tl.bfloat16).to(tl.float32)
 b0=lib.sqrt_rn(tl.sum((mean-mu)*(mean-mu),0));bp=tl.max(lib.sqrt_rn(tl.sum(v*v,1)),0)
 tl.store(MU+row*128+d,mu);tl.store(ST+(row*4+r)*12+11,tl.where(r==0,b0,tl.where(r==1,bp,0.)))
 raw=tl.load(RAW+(row*4+r[:,None])*128+d[None,:]);pk=raw-tl.sum(raw,0)[None,:]/4.+km[None,:];tl.store(PK+(row*4+r[:,None])*128+d[None,:],pk)
 pv=tl.load(PV+(row*4+r[:,None])*128+d[None,:]);basis=tl.load(BASIS+h*BH+d[:,None]*BD+rank[None,:]*BR).to(tl.bfloat16).to(tl.float32)
 code=tl.sum(pv[:,:,None]*basis[None,:,:],1);s0=tl.where(r<2,1.,-1.);s1=tl.where(r%2==0,1.,-1.);s2=tl.where((r==0)|(r==3),1.,-1.)
 tl.store(DETAIL+row*24+rank,tl.sum(code*s0[:,None],0)/4.);tl.store(DETAIL+row*24+8+rank,tl.sum(code*s1[:,None],0)/4.);tl.store(DETAIL+row*24+16+rank,tl.sum(code*s2[:,None],0)/4.)
@tr.jit
def _rep_stats(K,V,PK,MU,DETAIL,BASIS,ST,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
 rep=tl.program_id(0);page=rep//4;r=rep%4;h=page//P;p=page%P;t=tl.arange(0,16);d=tl.arange(0,128);rank=tl.arange(0,8)
 k=tl.load(K+h*KH+p*KP+(r*16+t[:,None])*KT+d[None,:]).to(tl.float32);v=tl.load(V+h*VH+p*VP+(r*16+t[:,None])*VT+d[None,:]).to(tl.float32)
 km=tl.sum(k,0)/16.;vm=tl.sum(v,0)/16.;kd=k-km[None,:];var128=tl.sum(kd*kd,0)/16.;var=tl.sum(tl.reshape(var128,(8,16)),1)/16.
 mu=tl.load(MU+page*128+d).to(tl.float32);basis=tl.load(BASIS+h*BH+d[:,None]*BD+rank[None,:]*BR).to(tl.bfloat16).to(tl.float32)
 d0=tl.load(DETAIL+page*24+rank).to(tl.float32);d1=tl.load(DETAIL+page*24+8+rank).to(tl.float32);d2=tl.load(DETAIL+page*24+16+rank).to(tl.float32)
 code=tl.where(r<2,1.,-1.)*d0+tl.where(r%2==0,1.,-1.)*d1+tl.where((r==0)|(r==3),1.,-1.)*d2
 reconstructed=mu+tl.sum(code[None,:]*basis,1);pk=tl.load(PK+rep*128+d).to(tl.float32);dk=km-pk;dv=vm-reconstructed;vd=v-vm[None,:]
 tl.store(ST+rep*12+rank,var);tl.store(ST+rep*12+8,lib.sqrt_rn(tl.sum(dk*dk,0)));tl.store(ST+rep*12+9,lib.sqrt_rn(tl.sum(dv*dv,0)));tl.store(ST+rep*12+10,lib.sqrt_rn(tl.sum(tl.sum(vd*vd,1),0)/16.))
@torch.no_grad()
def build(model,k,v,cheap=False,*,three_term_features=False):
 h,p,t,d=k.shape;assert (h,t,d)==(12,64,128) and k.dtype==v.dtype==torch.bfloat16 and k.stride(-1)==v.stride(-1)==1
 feat=torch.empty(h,p,64,32,device=k.device,dtype=torch.float32);raw=torch.empty(h,p,4,128,device=k.device);pv=torch.empty_like(raw)
 pk=torch.empty(h,p,4,128,device=k.device,dtype=torch.bfloat16);mu=torch.empty(h,p,128,device=k.device,dtype=torch.bfloat16);detail=torch.empty(h,p,3,8,device=k.device,dtype=torch.bfloat16);stats=torch.empty(h,p,4,12,device=k.device,dtype=torch.bfloat16)
 feature_kernel=_features_bf16 if three_term_features else _features
 feature_kernel[(h*p,)](k,v,model.bank.contiguous(),model.group_v.contiguous(),model.group_gate.contiguous(),feat,p,*k.stride()[:3],*v.stride()[:3],cheap,num_warps=4,num_stages=2,enable_fp_fusion=False)
 _pool_rep[(h*p*4,)](k,v,feat,model.pool_k.contiguous(),model.pool_v.contiguous(),model.pool_feature.contiguous(),raw,pv,p,*k.stride()[:3],*v.stride()[:3],num_warps=4,enable_fp_fusion=False)
 _page_summary[(h*p,)](k,v,raw,pv,model.basis,pk,mu,detail,stats,p,*k.stride()[:3],*v.stride()[:3],*model.basis.stride(),num_warps=4,enable_fp_fusion=False)
 _rep_stats[(h*p*4,)](k,v,pk,mu,detail,model.basis,stats,p,*k.stride()[:3],*v.stride()[:3],*model.basis.stride(),num_warps=4,enable_fp_fusion=False)
 return Summary(pk,mu,detail,stats)
