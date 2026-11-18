"""Two-kernel latest summary construction; same full-mean/Haar/risk formulas."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .ops import Summary
@tr.jit
def _pool(K,V,BANK,GV,GATE,WK,WV,WF,PK,PV,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,CHEAP:tl.constexpr):
 row=tl.program_id(0);h=row//P;p=row%P;t=tl.arange(0,64);d=tl.arange(0,128);f=tl.arange(0,32);r=tl.arange(0,4)
 k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.float32)
 v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.float32)
 km=tl.sum(k,0)/64.;bank=tl.load(BANK+h*32*128+f[None,:]*128+d[:,None])
 score=tl.dot(k-km[None,:],bank,input_precision='tf32x3')*0.08838834764831843
 rms=tl.maximum(lib.sqrt_rn(tl.sum(tl.sum(score*score,0),0)/2048.),1.e-6);feat=score/rms
 if not CHEAP:
  gg=tl.load(GV+h*128*32+d[:,None]*32+f[None,:]);vp=tl.dot(v,gg,input_precision='tf32x3')
  prob=tl.exp(score-tl.max(score,0)[None,:]);prob=prob/tl.sum(prob,0)[None,:]
  mean=tl.sum(prob*vp,0);resp=prob*64.*(vp-mean[None,:])
  rrms=tl.maximum(lib.sqrt_rn(tl.sum(tl.sum(resp*resp,0),0)/2048.),1.e-6)
  gate=.1*lib.tanh(tl.load(GATE+h*32+f));feat=feat+gate[None,:]*(resp/rrms)
 kg=tl.reshape(k,(4,16,128));vg=tl.reshape(v,(4,16,128));fg=tl.reshape(feat,(4,16,32))
 gkm=tl.sum(kg,1)/16.;gvm=tl.sum(vg,1)/16.
 wk=tl.load(WK+(h*4+r[:,None])*128+d[None,:]);wv=tl.load(WV+(h*4+r[:,None])*128+d[None,:]);wf=tl.load(WF+(h*4+r[:,None])*32+f[None,:])
 z=tl.sum((kg-gkm[:,None,:])*wk[:,None,:],2)+tl.sum((vg-gvm[:,None,:])*wv[:,None,:],2)
 z=z+tl.sum(fg*wf[:,None,:],2);z=2.*lib.tanh((z-tl.sum(z,1)[:,None]/16.)/2.)
 a=tl.exp(z-tl.max(z,1)[:,None]);a=a/tl.sum(a,1)[:,None]
 pk=tl.sum(a[:,:,None]*kg,1);pv=tl.sum(a[:,:,None]*vg,1)
 pk=pk-tl.sum(pk,0)[None,:]/4.+km[None,:]
 tl.store(PK+(row*4+r[:,None])*128+d[None,:],pk)
 tl.store(PV+(row*4+r[:,None])*128+d[None,:],pv)
@tr.jit
def _features(K,V,BANK,GV,GATE,FEAT,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,CHEAP:tl.constexpr,PROJECT:tl.constexpr=False,WF=None,SAVE_PAGE:tl.constexpr=False,PAGE_OUT=None):
 row=tl.program_id(0);h=row//P;p=row%P;t=tl.arange(0,64);d=tl.arange(0,128);f=tl.arange(0,32);r=tl.arange(0,4)
 k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.float32)
 v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.float32)
 km=tl.sum(k,0)/64.;bank=tl.load(BANK+h*32*128+f[None,:]*128+d[:,None])
 score=tl.dot(k-km[None,:],bank,input_precision='tf32x3')*0.08838834764831843
 rms=tl.maximum(lib.sqrt_rn(tl.sum(tl.sum(score*score,0),0)/2048.),1.e-6);feat=score/rms
 if not CHEAP:
  gg=tl.load(GV+h*128*32+d[:,None]*32+f[None,:]);vp=tl.dot(v,gg,input_precision='tf32x3')
  prob=tl.exp(score-tl.max(score,0)[None,:]);prob=prob/tl.sum(prob,0)[None,:]
  mean=tl.sum(prob*vp,0);resp=prob*64.*(vp-mean[None,:])
  rrms=tl.maximum(lib.sqrt_rn(tl.sum(tl.sum(resp*resp,0),0)/2048.),1.e-6)
  gate=.1*lib.tanh(tl.load(GATE+h*32+f));feat=feat+gate[None,:]*(resp/rrms)
 if SAVE_PAGE:
  tl.store(PAGE_OUT+row*257+d,km)
  tl.store(PAGE_OUT+row*257+128+d,tl.sum(v,0)/64.)
  tl.store(PAGE_OUT+row*257+256,tl.max(lib.sqrt_rn(tl.sum(v*v,1)),0))
 if PROJECT:
  wf=tl.load(WF+(h*4+t[:,None]//16)*32+f[None,:])
  tl.store(FEAT+row*64+t,tl.sum(feat*wf,1))
 else:
  tl.store(FEAT+(row*64+t[:,None])*32+f[None,:],feat)
@tr.jit
def _pool_features(K,V,FEAT,WK,WV,WF,PK,PV,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr):
 row=tl.program_id(0);h=row//P;p=row%P;t=tl.arange(0,64);d=tl.arange(0,128);f=tl.arange(0,32);r=tl.arange(0,4)
 k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.float32)
 v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.float32)
 km=tl.sum(k,0)/64.;feat=tl.load(FEAT+(row*64+t[:,None])*32+f[None,:])
 kg=tl.reshape(k,(4,16,128));vg=tl.reshape(v,(4,16,128));fg=tl.reshape(feat,(4,16,32))
 gkm=tl.sum(kg,1)/16.;gvm=tl.sum(vg,1)/16.
 wk=tl.load(WK+(h*4+r[:,None])*128+d[None,:]);wv=tl.load(WV+(h*4+r[:,None])*128+d[None,:]);wf=tl.load(WF+(h*4+r[:,None])*32+f[None,:])
 z=tl.sum((kg-gkm[:,None,:])*wk[:,None,:],2)+tl.sum((vg-gvm[:,None,:])*wv[:,None,:],2)
 z=z+tl.sum(fg*wf[:,None,:],2);z=2.*lib.tanh((z-tl.sum(z,1)[:,None]/16.)/2.)
 a=tl.exp(z-tl.max(z,1)[:,None]);a=a/tl.sum(a,1)[:,None]
 pk=tl.sum(a[:,:,None]*kg,1);pv=tl.sum(a[:,:,None]*vg,1)
 pk=pk-tl.sum(pk,0)[None,:]/4.+km[None,:]
 tl.store(PK+(row*4+r[:,None])*128+d[None,:],pk)
 tl.store(PV+(row*4+r[:,None])*128+d[None,:],pv)
@tr.jit
def _summary(K,V,PK,PV,BASIS,MU,DETAIL,ST,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr):
 row=tl.program_id(0);h=row//P;p=row%P;t=tl.arange(0,64);d=tl.arange(0,128);r=tl.arange(0,4);rank=tl.arange(0,8)
 k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.float32);v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.float32)
 mean=tl.sum(v,0)/64.;mu=mean.to(tl.bfloat16).to(tl.float32);tl.store(MU+row*128+d,mu)
 pv=tl.load(PV+(row*4+r[:,None])*128+d[None,:]);basis=tl.load(BASIS+h*BH+d[:,None]*BD+rank[None,:]*BR).to(tl.bfloat16).to(tl.float32)
 code=tl.sum(pv[:,:,None]*basis[None,:,:],1)
 s0=tl.where(r<2,1.,-1.);s1=tl.where(r%2==0,1.,-1.);s2=tl.where((r==0)|(r==3),1.,-1.)
 d0=(tl.sum(code*s0[:,None],0)/4.).to(tl.bfloat16).to(tl.float32);d1=(tl.sum(code*s1[:,None],0)/4.).to(tl.bfloat16).to(tl.float32);d2=(tl.sum(code*s2[:,None],0)/4.).to(tl.bfloat16).to(tl.float32)
 tl.store(DETAIL+row*24+rank,d0);tl.store(DETAIL+row*24+8+rank,d1);tl.store(DETAIL+row*24+16+rank,d2)
 mixed=s0[:,None]*d0[None,:]+s1[:,None]*d1[None,:]+s2[:,None]*d2[None,:]
 reconstructed=mu[None,:]+tl.sum(mixed[:,None,:]*basis[None,:,:],2)
 kg=tl.reshape(k,(4,16,128));vg=tl.reshape(v,(4,16,128));km=tl.sum(kg,1)/16.;vm=tl.sum(vg,1)/16.
 kd=kg-km[:,None,:];variance=tl.sum(kd*kd,1)/16.;var=tl.sum(tl.reshape(variance,(4,8,16)),2)/16.
 pk=tl.load(PK+(row*4+r[:,None])*128+d[None,:]).to(tl.float32);dk=km-pk;dv=vm-reconstructed;vd=vg-vm[:,None,:]
 bk=lib.sqrt_rn(tl.sum(dk*dk,1));bv=lib.sqrt_rn(tl.sum(dv*dv,1));sv=lib.sqrt_rn(tl.sum(tl.sum(vd*vd,2),1)/16.)
 b0=lib.sqrt_rn(tl.sum((mean-mu)*(mean-mu),0));bp=tl.max(lib.sqrt_rn(tl.sum(v*v,1)),0)
 tl.store(ST+(row*4+r[:,None])*12+rank[None,:],var)
 tl.store(ST+(row*4+r)*12+8,bk);tl.store(ST+(row*4+r)*12+9,bv);tl.store(ST+(row*4+r)*12+10,sv)
 tl.store(ST+(row*4+r)*12+11,tl.where(r==0,b0,tl.where(r==1,bp,0.)))
@torch.no_grad()
def build(model,k,v,cheap=False,*,pool_warps=4,summary_warps=4,split_features=False):
 h,p,t,d=k.shape;assert (h,t,d)==(12,64,128) and k.dtype==v.dtype==torch.bfloat16 and k.stride(-1)==v.stride(-1)==1
 pk=torch.empty(h,p,4,128,device=k.device,dtype=torch.bfloat16);pv=torch.empty(h,p,4,128,device=k.device,dtype=torch.float32)
 mu=torch.empty(h,p,128,device=k.device,dtype=torch.bfloat16);detail=torch.empty(h,p,3,8,device=k.device,dtype=torch.bfloat16);stats=torch.empty(h,p,4,12,device=k.device,dtype=torch.bfloat16)
 if split_features:
  feat=torch.empty(h,p,64,32,device=k.device,dtype=torch.float32)
  _features[(h*p,)](k,v,model.bank.contiguous(),model.group_v.contiguous(),model.group_gate.contiguous(),feat,p,*k.stride()[:3],*v.stride()[:3],cheap,num_warps=pool_warps,num_stages=2,enable_fp_fusion=False)
  _pool_features[(h*p,)](k,v,feat,model.pool_k.contiguous(),model.pool_v.contiguous(),model.pool_feature.contiguous(),pk,pv,p,*k.stride()[:3],*v.stride()[:3],num_warps=4,enable_fp_fusion=False)
 else:
  _pool[(h*p,)](k,v,model.bank.contiguous(),model.group_v.contiguous(),model.group_gate.contiguous(),model.pool_k.contiguous(),model.pool_v.contiguous(),model.pool_feature.contiguous(),pk,pv,p,*k.stride()[:3],*v.stride()[:3],cheap,num_warps=pool_warps,num_stages=2,enable_fp_fusion=False)
 _summary[(h*p,)](k,v,pk,pv,model.basis,mu,detail,stats,p,*k.stride()[:3],*v.stride()[:3],*model.basis.stride(),num_warps=summary_warps,enable_fp_fusion=False)
 return Summary(pk,mu,detail,stats)
