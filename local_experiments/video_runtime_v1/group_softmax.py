"""Preserve spatial-softmax's 32-thread reduction, distribute columns across CTAs."""
import ctypes,hashlib,subprocess
from filelock import FileLock
from functools import lru_cache
from pathlib import Path
import torch

@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/group_softmax.cu'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=Path('/tmp/streaming_sparse-0/group_softmax')/digest;folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_80','--shared','-Xcompiler','-fPIC',str(source),'-o',str(binary)],check=True)
    lib=ctypes.CDLL(str(binary));lib.group_softmax.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.group_softmax.restype=None
    lib.group_staged.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.group_staged.restype=None
    return lib

def softmax(x,columns=4,coalesced=True):
    assert x.is_cuda and x.dtype==torch.float32 and x.is_contiguous() and x.shape[-1]==32 and columns in (4,32)
    output=torch.empty_like(x)
    library().group_softmax(output.data_ptr(),x.data_ptr(),x.numel()//(x.shape[-2]*32),x.shape[-2],columns,int(coalesced),torch.cuda.current_stream().cuda_stream)
    return output

def staged(x,columns):
    assert x.is_cuda and x.is_contiguous() and x.dtype==torch.float32 and x.shape[-2:]==(4608,32) and columns in (1,2,4)
    output=torch.empty_like(x)
    library().group_staged(output.data_ptr(),x.data_ptr(),x.numel()//(4608*32),columns,torch.cuda.current_stream().cuda_stream)
    return output
