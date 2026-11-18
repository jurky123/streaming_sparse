"""Experimental pool/statistics reuse; original reductions and BF16 boundaries retained."""
import torch,triton as tr,triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from ..review_v1.fused_summary import _features
from ..review_v1.ops import Summary
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
def _pool_row_sum(x, WIDTH:tl.constexpr, FEATURE_LAYOUT:tl.constexpr):
 # Fold the row bits in the same order as the original four-warp reduction.
 v=tl.reshape(x,(2,2,2,2,WIDTH))
 if FEATURE_LAYOUT:
  v=tl.permute(v,(4,1,0,3,2))
 else:
  v=tl.permute(v,(4,2,1,3,0))
 a,b=tl.split(v);a,b=tl.split(a+b);a,b=tl.split(a+b);a,b=tl.split(a+b)
 return a+b

@tr.jit
def _pool_rep(K,V,F,WK,WV,WF,PK,PV,AUX,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,AUX_STRIDE:tl.constexpr=265,PROJECT:tl.constexpr=False,PARTIAL:tl.constexpr=False,MATCH_ROWS:tl.constexpr=False):
 rep=tl.program_id(0);page=rep//4;r=rep%4;h=page//P;p=page%P;t=tl.arange(0,16);d=tl.arange(0,128);f=tl.arange(0,32)
 k=tl.load(K+h*KH+p*KP+(r*16+t[:,None])*KT+d[None,:]).to(tl.float32);v=tl.load(V+h*VH+p*VP+(r*16+t[:,None])*VT+d[None,:]).to(tl.float32)
 wk=tl.load(WK+(h*4+r)*128+d);wv=tl.load(WV+(h*4+r)*128+d);wf=tl.load(WF+(h*4+r)*32+f)
 if PROJECT:
  feature_score=tl.load(F+page*64+r*16+t)
 else:
  feat=tl.load(F+(page*64+r*16+t[:,None])*32+f[None,:])
  feature_score=tl.sum(feat*wf[None,:],1)
 if MATCH_ROWS:
  km=_pool_row_sum(k,128,False)/16.;vm=_pool_row_sum(v,128,False)/16.
 else:
  km=tl.sum(k,0)/16.;vm=tl.sum(v,0)/16.
 z=tl.sum((k-km[None,:])*wk[None,:],1)+tl.sum((v-vm[None,:])*wv[None,:],1)
 z=z+feature_score
 center=tl.sum(z,0) if not MATCH_ROWS else tl.sum(_pool_row_sum(z,1,True),0)
 z=2.*lib.tanh((z-center/16.)/2.);a=tl.exp(z-tl.max(z,0))
 norm=tl.sum(a,0) if not MATCH_ROWS else tl.sum(_pool_row_sum(a,1,True),0)
 a=a/norm
 if MATCH_ROWS:
  pk=_pool_row_sum(a[:,None]*k,128,True);pv=_pool_row_sum(a[:,None]*v,128,True)
 else:
  pk=tl.sum(a[:,None]*k,0);pv=tl.sum(a[:,None]*v,0)
 tl.store(PK+rep*128+d,pk);tl.store(PV+rep*128+d,pv)
 kd=k-km[None,:];var128=(_pool_row_sum(kd*kd,128,False) if MATCH_ROWS else tl.sum(kd*kd,0))/16.;var=tl.sum(tl.reshape(var128,(8,16)),1)/16.
 vd=v-vm[None,:];norms=tl.sum(vd*vd,1);sv=lib.sqrt_rn((tl.sum(_pool_row_sum(norms,1,False),0) if MATCH_ROWS else tl.sum(norms,0))/16.)
 tl.store(AUX+rep*AUX_STRIDE+d,km);tl.store(AUX+rep*AUX_STRIDE+128+d,vm);tl.store(AUX+rep*AUX_STRIDE+256+tl.arange(0,8),var);tl.store(AUX+rep*AUX_STRIDE+264,sv)
 if PARTIAL:
  tl.store(AUX+rep*AUX_STRIDE+265,tl.max(lib.sqrt_rn(tl.sum(v*v,1)),0))
@tr.jit
def _page_summary(K,V,RAW,PV,BASIS,PK,MU,DETAIL,ST,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr,FUSE_STATS:tl.constexpr=False,AUX=None,AUX_STRIDE:tl.constexpr=265,PARTIAL:tl.constexpr=False,FROM_FEATURES:tl.constexpr=False,PAGE_IN=None,OUT_PAGES:tl.constexpr=0):
 row=tl.program_id(0);h=row//P;p=row%P;t=tl.arange(0,64);d=tl.arange(0,128);r=tl.arange(0,4);rank=tl.arange(0,8)
 dst=h*(OUT_PAGES if OUT_PAGES else P)+p
 if FROM_FEATURES:
  km=tl.load(PAGE_IN+row*257+d)
  mean=tl.load(PAGE_IN+row*257+128+d)
  bp=tl.load(PAGE_IN+row*257+256)
 elif PARTIAL:
  rep=row*4+r
  km=tl.sum(tl.load(AUX+rep[:,None]*AUX_STRIDE+d[None,:]),0)/4.
  mean=tl.sum(tl.load(AUX+rep[:,None]*AUX_STRIDE+128+d[None,:]),0)/4.
  bp=tl.max(tl.load(AUX+rep*AUX_STRIDE+265),0)
 else:
  k=tl.load(K+h*KH+p*KP+t[:,None]*KT+d[None,:]).to(tl.float32);km=tl.sum(k,0)/64.
  v=tl.load(V+h*VH+p*VP+t[:,None]*VT+d[None,:]).to(tl.float32);mean=tl.sum(v,0)/64.
  bp=tl.max(lib.sqrt_rn(tl.sum(v*v,1)),0)
 mu=mean.to(tl.bfloat16).to(tl.float32)
 b0=lib.sqrt_rn(tl.sum((mean-mu)*(mean-mu),0))
 tl.store(MU+dst*128+d,mu);tl.store(ST+(dst*4+r)*12+11,tl.where(r==0,b0,tl.where(r==1,bp,0.)))
 raw=tl.load(RAW+(row*4+r[:,None])*128+d[None,:]);pk=raw-tl.sum(raw,0)[None,:]/4.+km[None,:];tl.store(PK+(dst*4+r[:,None])*128+d[None,:],pk)
 pv=tl.load(PV+(row*4+r[:,None])*128+d[None,:]);basis=tl.load(BASIS+h*BH+d[:,None]*BD+rank[None,:]*BR).to(tl.bfloat16).to(tl.float32)
 code=tl.sum(pv[:,:,None]*basis[None,:,:],1);s0=tl.where(r<2,1.,-1.);s1=tl.where(r%2==0,1.,-1.);s2=tl.where((r==0)|(r==3),1.,-1.)
 tl.store(DETAIL+dst*24+rank,tl.sum(code*s0[:,None],0)/4.);tl.store(DETAIL+dst*24+8+rank,tl.sum(code*s1[:,None],0)/4.);tl.store(DETAIL+dst*24+16+rank,tl.sum(code*s2[:,None],0)/4.)

 if FUSE_STATS:
  codes0=(tl.sum(code*s0[:,None],0)/4.).to(tl.bfloat16).to(tl.float32)
  codes1=(tl.sum(code*s1[:,None],0)/4.).to(tl.bfloat16).to(tl.float32)
  codes2=(tl.sum(code*s2[:,None],0)/4.).to(tl.bfloat16).to(tl.float32)
  compact=s0[:,None]*codes0[None,:]+s1[:,None]*codes1[None,:]+s2[:,None]*codes2[None,:]
  rec=mu[None,:]+tl.sum(compact[:,None,:]*basis[None,:,:],2)
  rep=row*4+r
  dk=tl.load(AUX+rep[:,None]*AUX_STRIDE+d[None,:])-pk.to(tl.bfloat16).to(tl.float32)
  dv=tl.load(AUX+rep[:,None]*AUX_STRIDE+128+d[None,:])-rec
  var=tl.load(AUX+rep[:,None]*AUX_STRIDE+256+rank[None,:])
  sv=tl.load(AUX+rep*AUX_STRIDE+264)
  tl.store(ST+(dst*4+r[:,None])*12+rank[None,:],var)
  tl.store(ST+(dst*4+r)*12+8,lib.sqrt_rn(tl.sum(dk*dk,1)))
  tl.store(ST+(dst*4+r)*12+9,lib.sqrt_rn(tl.sum(dv*dv,1)))
  tl.store(ST+(dst*4+r)*12+10,sv)
@tr.jit
def _rep_stats(AUX,PK,MU,DETAIL,BASIS,ST,P:tl.constexpr,KH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,VH:tl.constexpr,VP:tl.constexpr,VT:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr,BR:tl.constexpr,AUX_STRIDE:tl.constexpr=265,OUT_PAGES:tl.constexpr=0):
 rep=tl.program_id(0);page=rep//4;r=rep%4;h=page//P;p=page%P;t=tl.arange(0,16);d=tl.arange(0,128);rank=tl.arange(0,8)
 dst=h*(OUT_PAGES if OUT_PAGES else P)+p
 dst_rep=dst*4+r
 km=tl.load(AUX+rep*AUX_STRIDE+d);vm=tl.load(AUX+rep*AUX_STRIDE+128+d);var=tl.load(AUX+rep*AUX_STRIDE+256+rank);sv=tl.load(AUX+rep*AUX_STRIDE+264)
 mu=tl.load(MU+dst*128+d).to(tl.float32);basis=tl.load(BASIS+h*BH+d[:,None]*BD+rank[None,:]*BR).to(tl.bfloat16).to(tl.float32)
 d0=tl.load(DETAIL+dst*24+rank).to(tl.float32);d1=tl.load(DETAIL+dst*24+8+rank).to(tl.float32);d2=tl.load(DETAIL+dst*24+16+rank).to(tl.float32)
 code=tl.where(r<2,1.,-1.)*d0+tl.where(r%2==0,1.,-1.)*d1+tl.where((r==0)|(r==3),1.,-1.)*d2
 reconstructed=mu+tl.sum(code[None,:]*basis,1);pk=tl.load(PK+dst_rep*128+d).to(tl.float32);dk=km-pk;dv=vm-reconstructed
 tl.store(ST+dst_rep*12+rank,var);tl.store(ST+dst_rep*12+8,lib.sqrt_rn(tl.sum(dk*dk,0)));tl.store(ST+dst_rep*12+9,lib.sqrt_rn(tl.sum(dv*dv,0)));tl.store(ST+dst_rep*12+10,sv)
@torch.no_grad()
def build(model,k,v,cheap=False,*,three_term_features=False,aligned_aux=False,project_features=False,fuse_stats=False,partial_page=False,feature_page=False,pool_warps=4,match_pool=False,cuda_pool=False,out=None):
 h,p,t,d=k.shape;aux_stride=272 if aligned_aux else 266 if partial_page else 265;aux=torch.empty(h*p*4,aux_stride,device=k.device,dtype=torch.float32);assert (h,t,d)==(12,64,128) and k.dtype==v.dtype==torch.bfloat16 and k.stride(-1)==v.stride(-1)==1
 feat=torch.empty(h,p,64,1 if project_features else 32,device=k.device,dtype=torch.float32);raw=torch.empty(h,p,4,128,device=k.device);pv=torch.empty_like(raw)
 if out is None:
  pk=torch.empty(h,p,4,128,device=k.device,dtype=torch.bfloat16);mu=torch.empty(h,p,128,device=k.device,dtype=torch.bfloat16);detail=torch.empty(h,p,3,8,device=k.device,dtype=torch.bfloat16);stats=torch.empty(h,p,4,12,device=k.device,dtype=torch.bfloat16)
  out_pages=0
 else:
  pk,mu,detail,stats=out.keys,out.mean,out.detail,out.stats
  out_pages=mu.stride(0)//128
 feature_kernel=_features_bf16 if three_term_features else _features
 feature_options=dict(PROJECT=True,WF=model.pool_feature.contiguous()) if project_features else {}
 if feature_page:
  page_out=torch.empty((h,p,257),device=k.device,dtype=torch.float32)
  feature_options.update(SAVE_PAGE=True,PAGE_OUT=page_out)
 feature_kernel[(h*p,)](k,v,model.bank.contiguous(),model.group_v.contiguous(),model.group_gate.contiguous(),feat,p,*k.stride()[:3],*v.stride()[:3],cheap,num_warps=4,num_stages=2,enable_fp_fusion=False,**feature_options)
 if cuda_pool:
  from .pool_representative import pool
  pool(k,v,feat,model.pool_k.contiguous(),model.pool_v.contiguous(),model.pool_feature.contiguous(),raw,pv,aux)
 else:_pool_rep[(h*p*4,)](k,v,feat,model.pool_k.contiguous(),model.pool_v.contiguous(),model.pool_feature.contiguous(),raw,pv,aux,p,*k.stride()[:3],*v.stride()[:3],PARTIAL=partial_page,MATCH_ROWS=match_pool,PROJECT=project_features,AUX_STRIDE=aux_stride,num_warps=pool_warps,enable_fp_fusion=False)
 final_options=dict(FUSE_STATS=fuse_stats,PARTIAL=partial_page,AUX=aux,AUX_STRIDE=aux_stride) if fuse_stats or partial_page else {}
 if feature_page:final_options.update(FROM_FEATURES=True,PAGE_IN=page_out)
 _page_summary[(h*p,)](k,v,raw,pv,model.basis,pk,mu,detail,stats,p,*k.stride()[:3],*v.stride()[:3],*model.basis.stride(),num_warps=4,enable_fp_fusion=False,OUT_PAGES=out_pages,**final_options)
 if not fuse_stats:_rep_stats[(h*p*4,)](aux,pk,mu,detail,model.basis,stats,p,*k.stride()[:3],*v.stride()[:3],*model.basis.stride(),AUX_STRIDE=aux_stride,OUT_PAGES=out_pages,num_warps=4,enable_fp_fusion=False)
 return Summary(pk,mu,detail,stats)
