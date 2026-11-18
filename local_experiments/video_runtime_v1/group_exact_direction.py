import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

from filelock import FileLock
import torch

@lru_cache(None)
def library(root=False):
    source=Path(__file__).parent/'csrc'/('group_root_direction.cu' if root else 'group_exact_direction.cu')
    folder=Path('/tmp/streaming_sparse-0/exact_direction')/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_80','--shared','-Xcompiler','-fPIC',
                            str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    if root:
        lib.root_direction.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p]
        lib.root_direction.restype=None
        return lib
    lib.exact_direction.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.exact_direction.restype=None
    lib.raw_direction.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.raw_direction.restype=None
    return lib

@lru_cache(None)
def enable_large(device):
    with torch.cuda.device(device):
        assert library().enable_large_direction()==0

def scores(x,axis=None):
    h,n,d=x.shape
    assert x.is_contiguous() and d==32 and n in (32,64,128,256,512)
    out=x.new_empty(h,n)
    if n==512:enable_large(x.device)
    if axis is None:
        library().raw_direction(x.data_ptr(),out.data_ptr(),h,n,torch.cuda.current_stream(x.device).cuda_stream)
    else:library().exact_direction(x.data_ptr(),axis.data_ptr(),out.data_ptr(),h,n,torch.cuda.current_stream(x.device).cuda_stream)
    return out

def root_scores(x,axis):
    g,n,d=x.shape
    assert x.is_contiguous() and d==32 and n in (1536,4608)
    scratch=x.new_empty((g,32,32))
    direction=x.new_empty((g,32))
    out=x.new_empty((g,n))
    library(root=True).root_direction(x.data_ptr(),axis.data_ptr(),scratch.data_ptr(),direction.data_ptr(),out.data_ptr(),g,n,torch.cuda.current_stream(x.device).cuda_stream)
    return out
