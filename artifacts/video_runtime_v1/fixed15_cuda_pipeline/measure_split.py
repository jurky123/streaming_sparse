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
from rows import guard
from split_tail import partial,merge_parts
split=int(os.environ['TAIL_SPLIT'])
pm=torch.empty(split,h,n,D,device='cuda');pc=torch.empty(split,h,n,8,device='cuda');state=torch.empty(split,h,n,2,device='cuda')
def tail_parts():
 return partial[(72,12,split)](q,keys,mean,detail,None,pm,pc,state,p,n,h,D,BQ=64,BR=64,BF16_AV=True,MERGE=True,HEAD_PAGES=hp,IDS=ids,EXACT_COUNT=count,SPLIT=split,num_warps=4,num_stages=2,enable_fp_fusion=False)
def merge():return merge_parts[(n//16,12)](oe,lse,pm,pc,state,basis,new,split,num_warps=4,enable_fp_fusion=False)
def run():
 exact();guard[(72,12)](q,k,v,ids,oe,lse,mass,p,count,num_warps=4,num_stages=2,enable_fp_fusion=False)
 tail_parts();merge()
def current_split():repack(p-72);run()
control();run();diff=new.float()-old.float()
r=dict(max_abs=float(diff.abs().max()),nmse=float(diff.square().sum()/old.float().square().sum()),bitwise=torch.equal(old.view(torch.uint8),new.view(torch.uint8)))
names=['production','packed','current72','partial','merge']
ts=times([timed(f) for f in (control,run,current_split,tail_parts,merge)])
row=dict(ms={key:median(x) for key,x in zip(names,ts)},contracts=r,samples=dict(zip(names,ts)))
(root/f'split_{split}_{p}.json').write_text(json.dumps(row,indent=2))
print(row['ms'],r,flush=True)
