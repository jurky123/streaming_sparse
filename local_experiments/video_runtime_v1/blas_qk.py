"""Route QK with native BF16 operands and scaled FP32 cuBLAS output."""
import ctypes as ct
from functools import lru_cache

import torch


@lru_cache(None)
def _gemm():
    lib=ct.CDLL('libcublas.so.12')
    fn=lib.cublasGemmStridedBatchedEx
    i,v,l=ct.c_int,ct.c_void_p,ct.c_longlong
    fn.argtypes=[v,i,i,i,i,i,v,v,i,i,l,v,i,i,l,v,v,i,i,l,i,i,i]
    fn.restype=i
    return fn


def qk(q,keys,out,algo=-1):
    h,m,d=q.shape
    n=keys.shape[2]
    alpha,beta=ct.c_float(d**-.5),ct.c_float(0.)
    status=_gemm()(torch.cuda.current_blas_handle(),1,0,n,m,d,ct.byref(alpha),
                   keys.data_ptr(),14,d,keys.stride(0),q.data_ptr(),14,d,q.stride(0),ct.byref(beta),
                   out.data_ptr(),0,n,out.stride(0),h,68,algo)
    if status:raise RuntimeError(f'cuBLAS route QK returned {status}')
