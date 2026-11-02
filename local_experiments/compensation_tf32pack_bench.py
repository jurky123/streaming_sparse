"""Actual closed-loop inputs, recorded GPU isolation, bounded kernel search."""
import gc,json,os,subprocess,time
from pathlib import Path
import torch
import triton
from .learned_cache.io import atomic,digest,RunLock,disk_guard
from .compensation_kernels import compensate as original,reference,_tail,_merge
from .compensation_tf32pack_kernels import mask,pack,tail,compensate,_tf32pack_tail as _fast_tail
ROOT=Path('artifacts/compensation_kernel_tf32pack_20261030')
def foreign():
    out=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    return [int(x) for x in out.splitlines() if x.strip().isdigit() and int(x)!=os.getpid()]

@torch.no_grad()
def capture():
    from .broad_common import Engine
    from .rollout_suite import generate_chunk
    from .compensation_runtime import CompensationRuntime,WrappedKernel
    engine=Engine();rt=CompensationRuntime(engine,'artifacts/compensation_probe_20261030/model.pt',mode='raw')
    emb=torch.load('artifacts/broad_coverage_20261030/embeddings/horse_field.pt',weights_only=True).cuda()
    class Done(Exception):pass
    class Capture(WrappedKernel):
        def __getitem__(self,grid):
            normal=super().__getitem__(grid)
            def run(*args,**kwargs):
                r=self.rt;l=r.active_layer
                if l==14 and r.engine.current['chunk']==5 and r.engine.current['step']==0:
                    self.base[grid](*args,**kwargs)
                    data=dict(q=args[0],pk=r.pk[l],code=r.codes[l],lut=args[5],exact=args[7],lse=args[6],basis=r.basis[l],center=r.centers[l],gamma=r.gamma[l])
                    atomic(ROOT/'input.pt',{k:v.cpu() for k,v in data.items()},True)
                    raise Done()
                normal(*args,**kwargs)
            return run
    rt.kernel=Capture(rt,rt.kernel.base)
    try:
        for c in range(6):generate_chunk(engine,emb,1101,c)
    except Done:pass
    finally:rt.close()
    assert (ROOT/'input.pt').exists()

def timing(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();values=[]
    for _ in range(5):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        for __ in range(12):fn()
        b.record();b.synchronize();values.append(a.elapsed_time(b)/12)
    return dict(median_ms=sorted(values)[2],min_ms=min(values),max_ms=max(values),samples=values)

@torch.no_grad()
def main():
    lock=RunLock(ROOT)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        identity=dict(code={s:digest(Path('local_experiments')/s) for s in ['compensation_tf32pack_kernels.py','compensation_tf32pack_bench.py']},model=digest('artifacts/compensation_probe_20261030/model.pt'))
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        if not (ROOT/'input.pt').exists():capture();gc.collect();torch.cuda.empty_cache()
        d={k:v.cuda() for k,v in torch.load(ROOT/'input.pt',weights_only=True).items()};q=d['q'];p=338;n=4608;h=12;dim=128
        args=[d[k] for k in ['q','pk','code','lut','exact','lse','basis','center','gamma']]
        baseline=original(*args,p);bits=mask(d['lut'],n);packed=pack(d['pk'],p)
        checkbits=bits.cpu().long()
        for head in range(12):
            for tileid in range(72):
                got=[i for i in range(p) if (checkbits[head,tileid,i//32].item()>>(i%32))&1]
                assert got==d['lut'][0,head,tileid,:72].cpu().tolist()
        # Reused buffers isolate kernel work from host allocator timings.
        out=torch.empty(h,n,8,device='cuda');lse=torch.empty(h,n,device='cuda')
        oldfn=lambda:_tail[(n//16,h)](q,d['pk'],d['code'],d['lut'],out,lse,p,338,n,h,dim,16,16,num_warps=4)
        before=foreign();old=timing(oldfn);print(json.dumps(dict(baseline_tail=old,foreign=before)),flush=True)
        configs=[(bq,16,w,1,False) for bq,w in [(16,4),(16,8),(32,4),(64,4)]]
        rows=[]
        for config in configs:
            name='_'.join(map(str,config));path=ROOT/f'bench_{name}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            try:
                ta,ls,compiled=tail(q,packed,d['code'],bits,p,config)
                torch.testing.assert_close(ta,baseline[1],atol=1e-4,rtol=1e-3)
                torch.testing.assert_close(ls,baseline[2],atol=3e-5,rtol=2e-5)
                fresh=d['exact'].clone();fa=args.copy();fa[4]=fresh
                predicted=compensate(*fa,p,config)[0]
                error=(predicted.float()-baseline[0].float()).square().sum()/baseline[0].float().square().sum()
                assert error<1e-5,error
                bq,bp,w,st,di=config
                fn=lambda:_fast_tail[(triton.cdiv(n,bq),h)](q,packed,d['code'],bits,out,lse,p,338,n,h,dim,bq,bp,di,num_warps=w,num_stages=st)
                load1=foreign();times=timing(fn);load2=foreign()
                row=dict(config=config,passed=True,output_nmse=error.item(),exact_output=torch.equal(predicted,baseline[0]),regs=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared,tail_timing=times,foreign=load1+load2)
                dest=ROOT/'compiled'/name;dest.mkdir(parents=True,exist_ok=True)
                for ext in ['ptx','cubin']:
                    val=compiled.asm[ext];(dest/('tail.'+ext)).write_bytes(val if isinstance(val,bytes) else val.encode())
            except Exception as exc:row=dict(config=config,passed=False,error=repr(exc))
            atomic(path,row);rows.append(row);print(json.dumps(row),flush=True);disk_guard(ROOT,50,.3)
        clean=[r for r in rows if r['passed'] and not r['foreign']]
        good=clean or [r for r in rows if r['passed']]
        assert good,'no numerically valid candidate'
        best=min(good,key=lambda r:r['tail_timing']['median_ms'])
        result=dict(baseline_tail=old,baseline_foreign=before,rows=rows,selected=best['config'],timing_clean=bool(clean) and not before,baseline_output_scope='one real horse_field layer14 chunk5 step0 closed-loop input; quality equivalence still requires full videos',input_sha256=digest(ROOT/'input.pt'))
        atomic(ROOT/'benchmark.json',result);print('SELECTED',best['config'],flush=True)
    finally:lock.close()
if __name__=='__main__':main()
