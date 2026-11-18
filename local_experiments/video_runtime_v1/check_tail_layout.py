"""Edit one shared-memory layout in captured TTGIR; keep arithmetic and reduction IR."""
import argparse,json,statistics
import torch,triton
from triton.compiler import compiler
# Triton 3.2's greedy prototype regex stalls on location-rich TTGIR.
compiler.prototype_pattern['ttgir']=r'^\s*tt\.func\s+(?:public\s+)?(@\w+)\((.*?)\)\s*attributes\s*\{[^}]*\}\s*\{'
compiler.arg_type_pattern['ttgir']=r'%\w+: (!tt\.ptr<[^>]+>)'
from .weights import ROOT
from ..review_v1.tuned_attention import _tail
from ..high_sparse_runtime import _budget_mask
@torch.no_grad()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['phase1','blocked','paired'],default='phase1');variant=parser.parse_args().variant
 torch.set_num_threads(4);root=ROOT/'artifacts/video_runtime_v1/tail_layout_v1';s=torch.load(ROOT/'artifacts/video_runtime_v1/tail_liveness_v1/input.pt',weights_only=True)
 q=s['q'].cuda();keys,mean,detail=[s['summary'][k].cuda().contiguous() for k in ('keys','mean','detail')];ids=s['ids'].cuda().contiguous()[None]
 bits=torch.empty(12,72,16,device='cuda',dtype=torch.uint32);_budget_mask[(72,12)](ids,bits,72,43,64,num_warps=4)
 def outputs():return torch.empty(12,4608,128,device='cuda'),torch.empty(12,4608,8,device='cuda'),torch.empty(12,4608,device='cuda')
 out={'jit':outputs(),'ir_control':outputs(),variant:outputs()};args=(q,keys,mean,detail,bits)
 compiled=_tail[(72,12)](*args,*out['jit'],432,4608,12,128,BQ=64,BR=64,BF16_AV=True,num_warps=4,num_stages=2,enable_fp_fusion=False)
 code=compiled.asm['ttgir'];(root/'baseline.ttgir').write_text(code)
 old='#shared3 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>'
 new=old.replace('perPhase = 4','perPhase = 1');assert code.count(old)==1
 if variant=='phase1':candidate=code.replace(old,new)
 else:
  before4='#blocked4 = #triton_gpu.blocked<{sizePerThread = [4, 1, 4], threadsPerWarp = [2, 16, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>'
  after4=before4.replace('[4, 1, 4]','[1, 1, 4]').replace('[2, 16, 1]','[4, 8, 1]')
  before7='#blocked7 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>'
  after7=before7.replace('[4, 4]','[1, 4]').replace('[2, 16]','[4, 8]')
  assert code.count(before4)==code.count(before7)==1
  if variant=='paired':
   after4=after4.replace('[1, 1, 4]','[1, 2, 4]')
   after7=after7.replace('[1, 4]','[1, 8]')
  candidate=code.replace(before4,after4).replace(before7,after7)
 candidate=candidate.replace('@_tail(', '@_tail_layout_'+variant+'(');path=root/(variant+'.ttgir');path.write_text(candidate)
 print('compile IR',flush=True)
 kernels={'jit':compiled,'ir_control':triton.compile(str(root/'baseline.ttgir'),options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False)),variant:triton.compile(str(path),options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False))}
 print('compiled',flush=True)
 graphs={};resources={};contracts={}
 for name,kernel in kernels.items():
  def run():
   if name=='jit':_tail[(72,12)](*args,*out[name],432,4608,12,128,BQ=64,BR=64,BF16_AV=True,num_warps=4,num_stages=2,enable_fp_fusion=False)
   else:kernel[(72,12,1)](*args,*out[name])
  run();torch.cuda.synchronize();resources[name]=dict(registers=kernel.n_regs,spills=kernel.n_spills,shared=kernel.metadata.shared)
  for _ in range(3):run()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):run()
  graphs[name]=g;contracts[name]=[dict(equal=torch.equal(x,y),max_abs=float((x-y).abs().max())) for x,y in zip(out['jit'],out[name])]
  (root/(name+'.ptx')).write_text(kernel.asm['ptx'])
 # Reach steady device clocks before comparing small differences.
 for _ in range(500):
  for g in graphs.values():g.replay()
 torch.cuda.synchronize()
 times={name:[] for name in graphs}
 for rep in range(50):
  names=list(graphs) if rep%2==0 else list(reversed(graphs))
  for name in names:
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(10):graphs[name].replay()
   b.record();b.synchronize();times[name].append(a.elapsed_time(b)/10)
 med={n:statistics.median(t) for n,t in times.items()};result=dict(warmup_replays_per_variant=500,contracts=contracts,resources=resources,median_ms=med,speedup=med['jit']/med[variant],times=times)
 (root/(variant+'_result.json')).write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='times'})
if __name__=='__main__':main()
