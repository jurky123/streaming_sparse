"""CUDA stable radix sort for the two largest grouping levels."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock


@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/group_radix.cu'
    folder=Path('/tmp/streaming_sparse-0/group_radix')/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(exist_ok=True,parents=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80',
                            '--shared','-Xcompiler','-fPIC',str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    lib.group_radix.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]
    lib.group_radix.restype=None
    return lib


def order(scores, output):
    groups, size = scores.shape
    library().group_radix(scores.data_ptr(), output.data_ptr(), groups, size,
                          torch.cuda.current_stream(scores.device).cuda_stream)
