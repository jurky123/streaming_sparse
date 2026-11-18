"""CUDA representative pooling for the 64-token summary builder."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock


@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/pool_representative.cu'
    folder=Path('/tmp/streaming_sparse-0/pool_rep')/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(exist_ok=True,parents=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80',
                            '--shared','-Xcompiler','-fPIC',str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    lib.run_pool.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*8+[ctypes.c_void_p]
    lib.run_pool.restype=None
    return lib


def pool(k,v,f,wk,wv,wf,pk,pv,aux):
    tensors=(k,v,f,wk,wv,wf,pk,pv,aux)
    assert all(x.dtype==torch.float32 for x in tensors[2:]) and aux.shape[-1]==265 and f.shape[-1]==32
    library().run_pool(*(x.data_ptr() for x in tensors),*k.shape[:2],*k.stride()[:3],*v.stride()[:3],
                       torch.cuda.current_stream(k.device).cuda_stream)
