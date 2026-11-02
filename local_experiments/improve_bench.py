"""Actual held-out validation QKV; same LUT kernel grid, route breakdown. No video claims."""
import json,os,statistics,subprocess
from pathlib import Path
import torch
from .broad_common import Engine
from .improve_runtime import FastRuntime
from .rollout_suite import generate_chunk
from .rollout_kernels import route_scores
from .learned_cache.io import atomic,digest,RunLock
ROOT=Path('artifacts/rollout_improve_20261030')
def foreign():
    out=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(x) for x in out.splitlines() if x.strip().isdigit() and int(x)!=os.getpid()]
def timing(fn):
    for _ in range(10):fn()
    values=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(20):fn()
        end.record();end.synchronize();values.append(start.elapsed_time(end)/20)
    return dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),samples_ms=values)
class BenchKernel:
    def __init__(self,rt):self.rt=rt;self.original=rt.kernel;self.done=False
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            compiled=self.original[grid](*args,**kwargs);c=self.rt.engine.current
            if self.done or c['layer']!=14 or c['chunk']!=6 or c['step']!=0:return compiled
            self.done=True;before=foreign();reference=args[7].clone();scratch=torch.empty_like(reference);inputs=list(args);inputs[7]=scratch;rows=[]
            for warps in [4,8]:
                for stages in [1,2,3,4]:
                    try:
                        fn=lambda:self.original[grid](*inputs,num_warps=warps,num_stages=stages)
                        item=fn();torch.cuda.synchronize()
                        exact=torch.equal(reference,scratch);error=((reference.float()-scratch.float()).square().sum()/reference.float().square().sum()).item()
                        row=dict(warps=warps,stages=stages,exact=exact,relative_mse=error,n_regs=getattr(item,'n_regs',None),n_spills=getattr(item,'n_spills',None),shared_bytes=getattr(item.metadata,'shared',None),**timing(fn))
                    except Exception as exc:row=dict(warps=warps,stages=stages,error=repr(exc))
                    rows.append(row)
            q=args[0];p=self.rt.n[14];pk=self.rt.pk[14][:,:p];alpha=self.rt.alpha[14]
            z=route_scores(q,pk,alpha);selected=z.topk(72,-1).indices;current=torch.arange(p,p+72,device=q.device).expand(12,72,72)
            route=dict(score=timing(lambda:route_scores(q,pk,alpha)),topk_sorted=timing(lambda:z.topk(72,-1)),topk_unsorted=timing(lambda:z.topk(72,-1,sorted=False)),
                sort144=timing(lambda:torch.cat((selected,current),-1).sort(-1)),sort72=timing(lambda:selected.sort(-1)),full_base=timing(lambda:super(FastRuntime,self.rt).route(14,q,p)),full_fast=timing(lambda:self.rt.route(14,q,p)))
            after=foreign();base=next(r for r in rows if r['warps']==4 and r['stages']==3)
            valid=[r for r in rows if r.get('exact') and 'median_ms' in r];best=min(valid,key=lambda r:r['median_ms'])
            promote=not before and not after and best['median_ms']<base['median_ms']*.95
            atomic(ROOT/'kernel_benchmark.json',dict(rows=rows,route=route,foreign_before=before,foreign_after=after,promoted=promote,choice=[best['warps'],best['stages']] if promote else [4,3],
                source=digest(Path(__file__)),note='One true validation layer/head batch, fixed mask. Shared-device timings are diagnostic only. Full-video equality still required.'))
            return compiled
        return call
@torch.no_grad()
def main():
    root=ROOT/'bench';lock=RunLock(root)
    try:
        if (ROOT/'kernel_benchmark.json').exists():return
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        engine=Engine();rt=FastRuntime(engine);wrapper=BenchKernel(rt);rt.kernel=wrapper
        emb=torch.load('artifacts/broad_coverage_20261030/embeddings/horse_field.pt',weights_only=True).cuda()
        try:
            for ch in range(7):generate_chunk(engine,emb,101,ch)
            assert wrapper.done
        finally:rt.close()
    finally:lock.close()
if __name__=='__main__':main()
