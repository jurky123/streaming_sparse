import sys, ctypes as C, os
from pathlib import Path
root=Path(__file__).parent
sys.path.insert(0,str(root.parent/'fixed15_baseline_v1'))
from bench import *
from local_experiments.video_runtime_v1.page_prefetch import coeff_tail
from tail import complete
from pack_kernel import pack_pages
from fragments import pack
from statistics import median
variant=os.getenv('EXACT_VARIANT','shared_overlap')
lib=C.CDLL('/tmp/jzk_fixed15_'+variant+'.so')
lib.run.argtypes=[C.c_void_p]*7+[C.c_int,C.c_void_p]
pk,pv=torch.empty_like(k),torch.empty_like(v)
oe=torch.empty_like(q,dtype=torch.float32)
lse,mass=[torch.empty(q.shape[:-1],device='cuda') for _ in range(2)]
def repack(start):return pack_pages[(p-start,h,16)](k,v,pk,pv,start)
repack(0)
for x,y,is_value in [(pk,k,False),(pv,v,True)]:
 assert torch.equal(x.flatten(),pack(y,is_value).flatten())
def exact():
 status=lib.run(*(x.data_ptr() for x in (q,pk,pv,ids,oe,lse,mass)),count,torch.cuda.current_stream().cuda_stream)
 assert status==0
def tail(kernel,out,partial):
 return kernel[(72,12)](q,keys,mean,detail,None,mass if partial else None,None,None,p,n,h,D,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=out,HEAD_PAGES=hp,K=k,V=v,IDS=ids,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=count,QK_SCALE=128**-.5,EXACT=oe if partial else None,ELSE=lse if partial else None,num_warps=4,num_stages=2,enable_fp_fusion=False)
def control():return tail(coeff_tail,old,False)
def finish():return tail(complete,new,True)
def candidate():exact();finish()
def current():repack(p-72);candidate()
def allpages():repack(0);candidate()
from local_experiments.review_v1.tuned_attention import _merge,_tail
from local_experiments.high_sparse_runtime import _budget_mask
import triton
tail_name=os.getenv('TAIL_VARIANT','warptail')
tail_lib=C.CDLL('/tmp/jzk_fixed15_'+tail_name+'.so')
tail_lib.run.argtypes=[C.c_void_p]*8+[C.c_int]*3+[C.c_void_p]
mu=torch.empty(h,n,D,device='cuda');code=torch.empty(h,n,8,device='cuda');lt=torch.empty(h,n,device='cuda')
mu0=torch.empty_like(mu);code0=torch.empty_like(code);lt0=torch.empty_like(lt)
bits=torch.empty(h,n//64,16,device='cuda',dtype=torch.uint32)
_budget_mask[(72,12)](ids,bits,72,count,triton.next_power_of_2(count),num_warps=4)
def t_ref():
 return _tail[(72,12)](q,keys,mean,detail,bits,mu0,code0,lt0,p,n,h,D,BQ=64,BR=64,BF16_AV=True,HEAD_PAGES=hp,num_warps=4,num_stages=2,enable_fp_fusion=False)
tiles=(p+15)//16
def pad(x):
 out=torch.zeros((12,tiles*16,*x.shape[2:]),device=x.device,dtype=x.dtype)
 out[:,:p]=x
 return out.reshape(12,tiles,16,*x.shape[2:])
kk=pad(keys).reshape(12,tiles,64,128)
mk=pad(mean);dk=pad(detail)
u,v0,l,j,b=torch.meshgrid(torch.arange(8,device='cuda'),torch.arange(4,device='cuda'),torch.arange(32,device='cuda'),torch.arange(2,device='cuda'),torch.arange(4,device='cuda'),indexing='ij')
col=16*v0+8*j+l//4
red=16*u+2*(l%4)+8*(b//2)+b%2
ks=kk[:,:,col,red].reshape(12,tiles,-1)
f,l,j,b=torch.meshgrid(torch.arange(8,device='cuda'),torch.arange(32,device='cuda'),torch.arange(2,device='cuda'),torch.arange(4,device='cuda'),indexing='ij')
ms=mk[:,:,2*(l%4)+8*(b//2)+b%2,16*f+8*j+l//4].reshape(12,tiles,-1)
a,l,b=torch.meshgrid(torch.arange(3,device='cuda'),torch.arange(32,device='cuda'),torch.arange(4,device='cuda'),indexing='ij')
ds=dk[:,:,2*(l%4)+8*(b//2)+b%2,a,l//4].reshape(12,tiles,-1)
summaries=torch.cat((ks,ms,ds),dim=2).contiguous()
assert summaries.shape==(12,tiles,1328*8)
def t_cuda():
 status=tail_lib.run(*(x.data_ptr() for x in (q,summaries,mean,detail,ids,mu,code,lt)),p,hp,count,torch.cuda.current_stream().cuda_stream)
 assert status==0
def contract(a,b):
 e=a.float()-b.float()
 return dict(nmse=float(e.square().sum()/a.float().square().sum()),max_abs=float(e.abs().max()))
from summary_fragments import encode,combine
summaries_ref=summaries
summaries=torch.empty_like(summaries_ref)
def encode_all():
 return encode[(12,tiles,42)](keys,mean,detail,summaries,p,hp,tiles,num_warps=4)
encode_all();assert torch.equal(summaries,summaries_ref)
from repair import retry
libf=C.CDLL('/tmp/jzk_fixed15_relax_all.so')
libf.run.argtypes=[C.c_void_p]*8+[C.c_int]*2+[C.c_void_p]
flags=torch.empty(12,72,device='cuda',dtype=torch.int32)
def fused():
 status=libf.run(*(x.data_ptr() for x in (q,pk,pv,ids,new,summaries,basis,flags)),p,count,torch.cuda.current_stream().cuda_stream)
 assert status==0,status
def guard():
 return retry[(72,12)](flags,q,k,v,ids,keys,mean,detail,basis,new,p,hp,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
def run():encode_all();fused();guard()
def live():repack(p-72);run()
control();live();contracts=contract(old,new);contracts.update(fallback_tiles=int(flags.sum()),changed=int((old!=new).sum()))
names=['production','with_summary_pack','current72','fused','guard']
ts=times([timed(f) for f in (control,run,live,fused,guard)])
row=dict(ms={key:median(x) for key,x in zip(names,ts)},contracts=contracts,samples=dict(zip(names,ts)))
(root/f'relax_all_{p}.json').write_text(json.dumps(row,indent=2))
print(row['ms'],contracts,flush=True)
