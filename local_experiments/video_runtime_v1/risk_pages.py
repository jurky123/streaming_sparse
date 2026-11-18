"""Risk reduction with adjacent pages assigned to adjacent lanes."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock


@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/risk_pages.cu'
    folder=Path('/tmp/streaming_sparse-0/risk_pages')/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(exist_ok=True,parents=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_80',
                            '--shared','-Xcompiler','-fPIC',str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    lib.run_pages.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*3+[ctypes.c_void_p]
    lib.run_pages.restype=None
    return lib


def run(q2,qn,z,norm,stats,theta,c,out):
    lib=library()
    lib.run_pages(*(x.data_ptr() for x in (q2,qn,z,norm,stats,theta,c,out)),
                  z.shape[0],z.shape[1],z.shape[3],torch.cuda.current_stream(z.device).cuda_stream)
