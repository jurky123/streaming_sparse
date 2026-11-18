"""Fixed D1536 LayerNorm plus frame modulation, with BF16 intermediate rounding."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from filelock import FileLock
import torch

@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/layernorm_mod.cu'
    folder=Path('/tmp/streaming_sparse-0/layernorm_mod')/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_80','--shared','-Xcompiler','-fPIC',str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    lib.layernorm_mod.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_float]+[ctypes.c_int]*3+[ctypes.c_void_p]
    lib.layernorm_mod.restype=None
    lib.residual_norm_mod.argtypes=[ctypes.c_void_p]*6+lib.layernorm_mod.argtypes[4:]
    lib.residual_norm_mod.restype=None
    lib.gated_residual_norm.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int,ctypes.c_float,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.gated_residual_norm.restype=None
    return lib

def apply(x,scale,shift,eps):
    y=torch.empty_like(x)
    n=x.shape[1]
    library().layernorm_mod(x.data_ptr(),scale.data_ptr(),shift.data_ptr(),y.data_ptr(),n,eps,n//scale.shape[1],scale.stride(1),shift.stride(1),torch.cuda.current_stream(x.device).cuda_stream)
    return y

def add_and_modulate(x,addend,scale,shift,eps):
    residual=torch.empty_like(x)
    normalized=torch.empty_like(x)
    n=x.shape[1]
    library().residual_norm_mod(x.data_ptr(),addend.data_ptr(),scale.data_ptr(),shift.data_ptr(),
                                residual.data_ptr(),normalized.data_ptr(),n,eps,n//scale.shape[1],
                                scale.stride(1),shift.stride(1),torch.cuda.current_stream(x.device).cuda_stream)
    return residual,normalized

def gated_and_normalize(x,addend,gate,norm):
    residual=torch.empty_like(x)
    normalized=torch.empty_like(x)
    n=x.shape[1]
    library().gated_residual_norm(x.data_ptr(),addend.data_ptr(),gate.data_ptr(),norm.weight.data_ptr(),
                                  norm.bias.data_ptr(),residual.data_ptr(),normalized.data_ptr(),n,
                                  norm.eps,n//gate.shape[1],gate.stride(1),torch.cuda.current_stream(x.device).cuda_stream)
    return residual,normalized
