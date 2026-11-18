from test import *
import hashlib, re, inspect
from local_experiments.video_runtime_v1 import page_prefetch as pipeline
from local_experiments.review_v1.tuned_attention import _tail
def loop_hash(text):
 lines=text.splitlines();i=next(i for i,l in enumerate(lines) if '= scf.for' in l);end=next(j for j in range(i+1,len(lines)) if 'scf.yield' in lines[j])
 names={};body=re.sub(r' loc\(#loc\d+\)', '', '\n'.join(lines[i+1:end]))
 body=re.sub(r'%\w+',lambda m:names.setdefault(m[0],'%'+str(len(names))),body)
 return hashlib.sha256(body.encode()).hexdigest()
def main():
 from statistics import median
 b=base()
 digest=loop_hash(b.asm['ttgir'])
 print('hash',digest,flush=True)
 lines=b.asm['ttgir'].splitlines()
 start=next(i for i,l in enumerate(lines) if '= scf.for' in l)
 end=next(i for i in range(start+1,len(lines)) if 'scf.yield' in lines[i])
 body=lines[start+1:end]
 assert sum('async_commit_group' in l for l in body)==2 and sum('async_wait' in l for l in body)==1
 assert next(i for i,l in enumerate(body) if 'arith.addi' in l)>max(i for i,l in enumerate(body) if 'tt.dot ' in l)
 assert digest=="c1a0f3382ec80165c9013ac851c00d064cacbb59713a9956fb8069500d0a1bec"
 def launch():return pipeline.coeff_tail[(72,12)](q,keys,mean,detail,None,None,None,None,p,n,h,D,BQ=64,BR=64,BF16_AV=True,MERGE=True,BASIS=basis,OUT=new,HEAD_PAGES=hp,K=k,V=v,IDS=ids,ANCHOR_SCALE=True,COLUMN_MASS=True,EXACT_COUNT=count,QK_SCALE=128**-.5,num_warps=4,num_stages=2,enable_fp_fusion=False)
 ker=launch();assert '%waitv =' in ker.asm['ttgir']
 if args.profile:
  Path("/tmp/jzk_dep_fixed15.cubin").write_bytes(ker.asm["cubin"])
  for _ in range(50):launch()
  torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();launch();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop();return
 ts=times([timed(base),timed(launch)])
 delta=new.float()-old.float()
 row=dict(hash=digest,speedup=median(x/y for x,y in zip(*ts)),ms=list(map(median,ts)),bitwise=torch.equal(old.view(torch.uint8),new.view(torch.uint8)),max_abs=float(delta.abs().max()),spills=ker.n_spills,samples=ts)
 (root/f'pipeline_{p}.json').write_text(json.dumps(row,indent=2));print(str(row)[:210])
if __name__=='__main__':main()
