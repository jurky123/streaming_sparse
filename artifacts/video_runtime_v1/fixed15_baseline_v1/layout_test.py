from bench import *
from triton.compiler import compiler
from local_experiments.video_runtime_v1.page_prefetch import align_coefficients
import triton
compiler.prototype_pattern['ttgir']=r'^\s*tt\.func\s+(?:public\s+)?(@\w+)\((.*?)\)\s*attributes\s*\{[^}]*\}\s*\{'
compiler.arg_type_pattern['ttgir']=r'%\w+: (!tt\.ptr<[^>]+>)'
b=base()
from triton.runtime.driver import driver
source='paired_compiler.ttgir' if args.paired_stages==2 else f'paired_{p}_s1.ttgir'
text=(d/source).read_text();assert 'warpsPerCTA = [8, 1, 1], instrShape' in text
path=d/f'paired_layout_{p}_s{args.paired_stages}.ttgir'
text=text.replace('warpsPerCTA = [8, 1, 1], instrShape','warpsPerCTA = [2, 1, 4], instrShape')
import re
lines=text.splitlines()
for i,line in enumerate(lines):
 m=re.match(r'(\s*)(%\w+) = triton_gpu.convert_layout (%\w+) : (tensor<2x64x64xbf16, #mma1>) -> (tensor<.*opIdx = 0, parent = #mma1.*?>>)( loc.*)',line)
 if m:
  indent,dst,src,ty,target,loc=m.groups()
  mem="!tt.memdesc<2x64x64xbf16, #shared1, #triton_gpu.shared_memory>"
  buf=f'%abuf{i}'
  lines[i]=f'{indent}{buf} = triton_gpu.local_alloc {src} : ({ty}) -> {mem}{loc}\n{indent}{dst} = triton_gpu.local_load {buf} : {mem} -> {target}{loc}'
path.write_text('\n'.join(lines)+'\n')
ker=triton.compile(str(path),options=dict(num_warps=8,num_stages=args.paired_stages,enable_fp_fusion=False))
def launch():ker[(36,12,1)](q,keys,mean,detail,basis,new,k,v,ids)
launch();ts=times([timed(base),timed(launch)])
delta=new.float()-old.float()
r=dict(ms=list(map(median,ts)),speedup=median(x/y for x,y in zip(*ts)),bitwise=torch.equal(old.view(torch.uint8),new.view(torch.uint8)),nmse=float(delta.square().sum()/old.float().square().sum()),max_abs=float(delta.abs().max()),regs=ker.n_regs,spills=ker.n_spills,shared=ker.metadata.shared)
print(r);path.with_suffix('.json').write_text(json.dumps(r,indent=2))
