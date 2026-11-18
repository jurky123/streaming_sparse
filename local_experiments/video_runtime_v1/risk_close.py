"""Collect close Risk rows for selective FP32 recomputation."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock


@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/risk_close.cu'
    folder=Path('/tmp/streaming_sparse-0/risk_close')/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(exist_ok=True,parents=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80',
                            '--shared','-Xcompiler','-fPIC',str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    lib.mark_close.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.mark_close.restype=None
    lib.select_close.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.select_close.restype=None
    return lib


def mark(scores,own,rows,count):
    h,t,p=scores.shape
    library().mark_close(scores.data_ptr(),own.data_ptr(),rows.data_ptr(),count.data_ptr(),
                         h*t,p,torch.cuda.current_stream(scores.device).cuda_stream)


def select_close(scores,own,ids,mask,rows,count):
    h,t,p=scores.shape
    library().select_close(*(x.data_ptr() for x in (scores,own,ids,mask,rows,count)),h*t,p,
                           torch.cuda.current_stream(scores.device).cuda_stream)
