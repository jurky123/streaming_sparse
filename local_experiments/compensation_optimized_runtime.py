"""Same cached model and mask, optimized compensation backend; isolated from old suite."""
import torch
from .compensation_runtime import CompensationRuntime,WrappedKernel
from .compensation_kernels import reference
MODEL='artifacts/compensation_probe_20261030/model.pt'
class OptimizedKernel(WrappedKernel):
    def __getitem__(self,grid):
        def run(*args,**kwargs):
            self.base[grid](*args,**kwargs)
            r=self.rt;l=r.active_layer
            if not r.enabled[l]:return
            q=args[0];lut=args[5];lse=args[6];out=args[7];p=r.n[l]
            audit=r.comp_audit and r.engine.current['chunk'] in [2,5,20,40] and r.engine.current['step'] in [0,3]
            if audit:
                sl=slice(35*64,36*64);orig=out[:,sl].clone()
            result,_,_=r.optimized(q,r.pk[l],r.codes[l],lut,out,lse,r.basis[l],r.centers[l],r.gamma[l],p,r.config)
            assert result.data_ptr()==out.data_ptr()
            if audit:
                expected,_,_=reference(q[:,sl].contiguous(),r.pk[l],r.codes[l],lut[:,:,35:36].contiguous(),orig,lse[:,:,sl].contiguous(),r.basis[l],r.centers[l],r.gamma[l],p)
                error=(out[:,sl].float()-expected.float()).square().sum()/expected.float().square().sum().clamp_min(1e-8)
                assert error<1e-5 and torch.isfinite(out).all(),(l,p,error.item())
                r.comp_checks.append(dict(layer=l,chunk=r.engine.current['chunk'],step=r.engine.current['step'],nmse=error.item()))
        return run
class OptimizedCompensationRuntime(CompensationRuntime):
    def __init__(self,engine,backend,config,audit=False):
        super().__init__(engine,MODEL,mode='raw',audit=audit);self.method='rank8_'+backend;self.config=tuple(config)
        if backend=='fast':from .compensation_fast_kernels import compensate
        elif backend=='split':from .compensation_split_kernels import compensate
        elif backend=='packed':from .compensation_packed_kernels import compensate
        else:raise ValueError(backend)
        self.optimized=compensate;self.kernel=OptimizedKernel(self,self.kernel.base)
