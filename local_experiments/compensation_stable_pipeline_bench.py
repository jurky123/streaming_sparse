"""Full call costs, with exact output restored outside each timed interval."""
import importlib,json
from pathlib import Path
import torch
from .compensation_split_bench import foreign
from .learned_cache.io import atomic,digest,RunLock
ROOT=Path('artifacts/compensation_stable_pipeline_20261030')
@torch.no_grad()
def main():
    lock=RunLock(ROOT)
    try:
        torch.set_num_threads(4)
        source=Path('artifacts/compensation_kernel_opt_20261030/input.pt')
        d={k:v.cuda() for k,v in torch.load(source,weights_only=True).items()}
        args=[d[k] for k in ['q','pk','code','lut','exact','lse','basis','center','gamma']]
        exact=args[4].clone();rows=[]
        configs=[('original',None),('stable',(64,16,4,1,False))]
        for backend,config in configs:
            mod=importlib.import_module('.compensation_'+('kernels' if backend=='original' else backend+'_kernels'),__package__)
            def call():
                if backend=='original':
                    result=mod.compensate(*args,338);args[4].copy_(result[0])
                else:mod.compensate(*args,338,config)
            for _ in range(4):args[4].copy_(exact);call()
            torch.cuda.synchronize();load1=foreign();times=[]
            for _ in range(30):
                args[4].copy_(exact)
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();call();b.record();b.synchronize();times.append(a.elapsed_time(b))
            load2=foreign()
            if backend=='original':ref=args[4].clone()
            error=((args[4].float()-ref.float()).square().sum()/ref.float().square().sum()).item()
            assert error<1e-5
            row=dict(backend=backend,config=config,total_ms=sorted(times)[len(times)//2],min_ms=min(times),max_ms=max(times),samples=times,foreign=load1+load2,output_nmse=error)
            rows.append(row);atomic(ROOT/'results.json',dict(rows=rows,input_sha256=digest(source),scope='full compensation call including allocation, bitset, packing if used, tail, merge, original copy; reset excluded; steady p338; one actual input'))
            print(json.dumps(row),flush=True)
    finally:lock.close()
if __name__=='__main__':main()
