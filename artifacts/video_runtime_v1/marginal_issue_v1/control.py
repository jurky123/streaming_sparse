"""Rebuild the preceding deployment from the saved source for paired measurements."""
import ctypes
import subprocess
import hashlib
from filelock import FileLock
from pathlib import Path
from functools import lru_cache
from local_experiments.video_runtime_v1.fixed15_attention import library as deployed

@lru_cache(None)
def library():
    source = Path(__file__).with_name('control.cu')
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder = Path('/tmp/streaming_sparse-0/fixed15_attention') / key
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / 'kernel.so'
    with FileLock(str(folder / 'build.lock')):
        if not binary.exists():
            tmp = binary.with_suffix('.tmp.so')
            subprocess.run(['nvcc', '-O3', '--fmad=false', '-std=c++17', '-arch=sm_80',
                            '-shared', '-Xcompiler=-fPIC', str(source), '-o', str(tmp)], check=True)
            tmp.replace(binary)
    lib = ctypes.CDLL(str(binary))
    for name in ('run','prepare_fixed15'):
        getattr(lib,name).argtypes = getattr(deployed(),name).argtypes
    return lib
