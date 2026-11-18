from pathlib import Path
root=Path(__file__).parent
s=(root/'tail.py').read_text().replace('def complete','def partial').replace('COLUMN_MASS:tl.constexpr=False):','COLUMN_MASS:tl.constexpr=False,SPLIT:tl.constexpr=2):')
i=s.index(' if EXACT_COUNT:');j=s.index(' mx=',i)
s=s[:i]+' part=tl.program_id(2)\n'+s[j:]
s=s.replace('for first in range(tr.cdiv(P*4,BR)):','for first in range(part*tr.cdiv(tr.cdiv(P*4,BR),SPLIT),tl.minimum((part+1)*tr.cdiv(tr.cdiv(P*4,BR),SPLIT),tr.cdiv(P*4,BR))):')
s=s[:s.index(' safe=tl.maximum')]+''' row=(part*H+h)*N+qi
 tl.store(MU+row[:,None]*D+d[None,:],am)
 tl.store(CODE+row[:,None]*8+rank[None,:],ac,rank[None,:]<8)
 tl.store(LSE+2*row,mx);tl.store(LSE+2*row+1,den)
'''
s+='''
@tr.jit
def merge_parts(EX,LE,MU,AC,STATE,BASIS,OUT,S:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);q=t*16+tl.arange(0,16);d=tl.arange(0,128);rank=tl.arange(0,16)
 row=h*4608+q
 mx=tl.load(STATE+2*row)
 for i in tl.static_range(1,S):
  mx=tl.maximum(mx,tl.load(STATE+2*(row+i*12*4608)))
 am=tl.full((16,128),0.,tl.float32);ac=tl.full((16,16),0.,tl.float32);den=tl.full((16,),0.,tl.float32)
 for i in tl.static_range(S):
  r=row+i*12*4608
  w=tl.exp2(tl.load(STATE+2*r)-mx)
  am+=w[:,None]*tl.load(MU+r[:,None]*128+d[None,:])
  ac+=w[:,None]*tl.load(AC+r[:,None]*8+rank[None,:],rank[None,:]<8,0)
  den+=w*tl.load(STATE+2*r+1)
 safe=tl.maximum(den,1.e-30)
 basis=tl.load(BASIS+(h*128+d[None,:])*8+rank[:,None],rank[:,None]<8,0).to(tl.float32)
 tail=am/safe[:,None]+tl.dot(ac/safe[:,None],basis,input_precision='tf32x3')
 lt=tl.where(den>0,mx+tl.log2(safe)+4.,-float('inf'))
 le=tl.load(LE+q*12+h);rho=1./(1.+tl.exp2(le-lt))
 e=tl.load(EX+(q[:,None]*12+h)*128+d[None,:])
 tl.store(OUT+(q[:,None]*12+h)*128+d[None,:],e*(1.-rho[:,None])+tail*rho[:,None])
'''
(root/'split_tail.py').write_text(s)
s=(root/'full.py').read_text()
s=s[:s.index('c=control()')]+'''from rows import guard
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
'''
(root/'measure_split.py').write_text(s)
