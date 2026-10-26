"""Existing FlashInfer/FlexAttention, exact chunk-tail masks; no tail exclusion."""
import sys,torch,triton as tr
import triton.language as tl
from .setup import BASE
from ..opt19.library import _query,_indices
from ..opt14.flex_probe_gpu_meta import _transpose
from ..opt16.grouping import restore
@tr.jit
def _pack(K,V,KO,VO,N:tl.constexpr,B:tl.constexpr,SEG:tl.constexpr,PAGES:tl.constexpr):
 p=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,64);d=tl.arange(0,128);i=p//PAGES*SEG+p%PAGES*64+t
 valid=(i<N)&(p%PAGES*64+t<SEG);dst=(h*B+p)*64*128+t[:,None]*128+d[None,:]
 tl.store(KO+dst,tl.load(K+(h*N+i[:,None])*128+d[None,:],valid[:,None],0))
 tl.store(VO+dst,tl.load(V+(h*N+i[:,None])*128+d[None,:],valid[:,None],0))
def pack(kv,seg):
 h,n,d=kv.k.shape;b=kv.state['key'].shape[1];k=torch.empty((h,b*64,128),device=kv.k.device,dtype=kv.k.dtype);v=torch.empty_like(k)
 _pack[(b,h)](kv.k,kv.v,k,v,n,b,seg,tr.cdiv(seg,64),num_warps=4)
 return k,v
@tr.jit
def _lengths(IDS,O,T:tl.constexpr,K:tl.constexpr,PAGES:tl.constexpr,TAIL:tl.constexpr):
 z=tl.program_id(0)*256+tl.arange(0,256);h=tl.program_id(1);page=tl.load(IDS+h*T*K+z,z<T*K,0)
 tl.store(O+h*T*K+z,tl.where(page%PAGES==PAGES-1,TAIL,64),z<T*K)
VARIANT=r'''
struct ChunkTailAttention : AttentionVariantBase {
 static constexpr bool use_softmax = true;
 int32_t* lengths;uint32_t qo_len,kv_len,window_left;float sm_scale_log2;
 template<typename Params>
 __device__ __host__ ChunkTailAttention(const Params& params,uint32_t batch_idx,uint8_t* smem_ptr){
  qo_len=params.get_qo_len(batch_idx);kv_len=params.get_kv_len(batch_idx);window_left=kv_len;
  lengths=params.lengths+batch_idx*(kv_len/64);sm_scale_log2=0.08838834764831845f*math::log2e;
 }
 REGISTER_LOGITS_MASK(params,batch_idx,qo_idx,kv_idx,qo_head_idx,kv_head_idx,{
  return qo_idx<qo_len && kv_idx<kv_len && (kv_idx%64)<lengths[kv_idx/64];
 })
};
'''
class FlashInfer:
 def __init__(self,h,n,b,group,keep,seg):
  deps=BASE/'streaming_sparse/v10/experiments/opt19_tile_kernels_20261025/deps';sys.path.insert(0,str(deps));import flashinfer
  self.version=flashinfer.__version__;self.h=h;self.n=n;self.b=b;self.group=group;self.seg=seg;self.t=tr.cdiv(n,group);self.keep=keep;batch=h*self.t
  self.ids=torch.zeros((batch*keep,),device='cuda',dtype=torch.int32);self.lengths=torch.full_like(self.ids,64)
  qo=torch.arange(batch+1,device='cuda',dtype=torch.int32)*group;ki=torch.arange(batch+1,device='cuda',dtype=torch.int32)*keep;last=torch.full((batch,),64,device='cuda',dtype=torch.int32)
  jit=('opt24_chunk_tail_bf16',torch.bfloat16,torch.bfloat16,torch.bfloat16,torch.int32,128,128,['lengths'],['int32_t'],[],[],'ChunkTailAttention',VARIANT)
  self.wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8),kv_layout='NHD',backend='fa2',use_cuda_graph=False,jit_args=jit)
  self.wrapper.plan(qo,ki,self.ids,last,1,1,128,64,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16)
  assert self.ids.data_ptr()==self.wrapper._paged_kv_indices_buf.data_ptr()
 def prepared(self,q,k,v,ids,ix):
  nn=self.t*self.group;qp=torch.empty((self.h,nn,128),device=q.device,dtype=q.dtype)
  _query[(tr.cdiv(nn*128,256),self.h)](q,qp,self.n,nn,num_warps=4)
  _indices[(tr.cdiv(self.t*self.keep,256),self.h)](ids,self.ids,self.t,self.keep,self.b,ids.shape[1],1,num_warps=4)
  _lengths[(tr.cdiv(self.t*self.keep,256),self.h)](ids,self.lengths,self.t,self.keep,tr.cdiv(self.seg,64),self.seg%64 or 64,num_warps=4)
  y=self.wrapper.run(qp.reshape(self.h*nn,1,128),(k.reshape(self.h*self.b,64,1,128),v.reshape(self.h*self.b,64,1,128)),self.lengths).reshape(self.h,nn,128)[:,:self.n]
  return restore(y,ix)
 def run(self,q,kv,ids,ix):return self.prepared(q,*pack(kv,self.seg),ids,ix)

@tr.jit
def _split(IDS,P,F,PC,FC,PU,FU,T:tl.constexpr,K:tl.constexpr,B:tl.constexpr,PAGES:tl.constexpr,KK:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);j=tl.arange(0,KK);p=tl.load(IDS+(h*T+t)*K+j,j<K,0)
 partial=(j<K)&(p%PAGES==PAGES-1);full=(j<K)&(~partial)
 rp=tl.cumsum(partial.to(tl.int32),0)-1;rf=tl.cumsum(full.to(tl.int32),0)-1
 tl.store(P+(h*T+t)*K+rp,p,partial);tl.store(F+(h*T+t)*K+rf,p,full)
 tl.store(PC+h*T+t,tl.sum(partial.to(tl.int32),0));tl.store(FC+h*T+t,tl.sum(full.to(tl.int32),0))
 tl.store(PU+(h*T+t)*B+p,True,partial);tl.store(FU+(h*T+t)*B+p,True,full)
class Flex:
 def __init__(self,h,n,b,group,keep,seg,warps=4,stages=3):
  from torch.nn.attention.flex_attention import flex_attention
  import importlib
  torch.compile(lambda x:x+1)(torch.ones(1,device='cuda'))
  impl=importlib.import_module('torch._inductor.kernel.flex_attention');impl._get_default_config_fwd=lambda query:(group,64,warps,stages)
  torch._dynamo.config.cache_size_limit=128
  self.compiled=torch.compile(flex_attention,dynamic=False);self.group=group;self.seg=seg;self.version=torch.__version__;self.options={'BLOCK_M':group,'BLOCK_N':64,'ROWS_GUARANTEED_SAFE':True}
 def prepared(self,q,k,v,ids,ix):
  from torch.nn.attention.flex_attention import BlockMask
  assert not q.requires_grad and not k.requires_grad
  h,n,d=q.shape;t=ids.shape[1];keep=ids.shape[2];b=k.shape[1]//64;pages=tr.cdiv(self.seg,64);tail=self.seg%64 or 64
  pi=torch.empty_like(ids);fi=torch.empty_like(ids);pc=torch.empty((h,t),device=q.device,dtype=torch.int32);fc=torch.empty_like(pc)
  pu=torch.zeros((h,t,b),device=q.device,dtype=torch.bool);fu=torch.zeros_like(pu)
  _split[(t,h)](ids,pi,fi,pc,fc,pu,fu,t,keep,b,pages,tr.next_power_of_2(keep),num_warps=4)
  pqi=torch.empty((h,b,t),device=q.device,dtype=torch.int32);fqi=torch.empty_like(pqi);pqc=torch.empty((h,b),device=q.device,dtype=torch.int32);fqc=torch.empty_like(pqc)
  _transpose[(b,h)](pu,pqi,pqc,t,b,tr.next_power_of_2(t),num_warps=4);_transpose[(b,h)](fu,fqi,fqc,t,b,tr.next_power_of_2(t),num_warps=4)
  def visible(batch,head,qi,ki):return (ki//64%pages!=pages-1)|(ki%64<tail)
  bm=BlockMask((n,b*64),pc[None],pi[None],fc[None],fi[None],pqc[None],pqi[None],fqc[None],fqi[None],(self.group,64),visible)
  y=self.compiled(q[None],k[None],v[None],block_mask=bm,kernel_options=self.options)[0]
  return restore(y,ix)
 def run(self,q,kv,ids,ix):return self.prepared(q,*pack(kv,self.seg),ids,ix)
