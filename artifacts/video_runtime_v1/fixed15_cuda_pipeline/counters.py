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
which=os.environ['PROFILE_STAGE']
f={'production':control,'exact':exact,'tail':finish,'pack_current':lambda:repack(p-72)}[which]
exact()
for _ in range(20):f()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart();f();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
