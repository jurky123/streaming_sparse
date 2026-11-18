"""SM80 FFN projection with the native BF16 rounding and GELU values."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

from filelock import FileLock
import torch

from .weights import ROOT


@lru_cache(None)
def library():
    source=Path(__file__).parent/'csrc/ffn_rounded_gelu.cu'
    include=ROOT/'external_baselines/Sparse-VideoGen/svg/kernels/3rdparty/flashinfer/3rdparty/cutlass/include'
    digest=hashlib.sha256(source.read_bytes()+(include/'cutlass/version.h').read_bytes()).hexdigest()[:16]
    folder=Path('/tmp/streaming_sparse-0/ffn_rounded_gelu')/digest
    folder.mkdir(parents=True,exist_ok=True)
    binary=folder/'kernel.so'
    with FileLock(str(folder/'build.lock')):
        if not binary.exists():
            tmp=binary.with_suffix('.tmp.so')
            subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_80','-std=c++17',
                            '--expt-relaxed-constexpr','-I'+str(include),'--shared','-Xcompiler','-fPIC',
                            str(source),'-o',str(tmp)],check=True)
            tmp.replace(binary)
    lib=ctypes.CDLL(str(binary))
    lib.set_gelu_table.argtypes=[ctypes.c_void_p]
    lib.ffn_gelu.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*5
    return lib


@lru_cache(None)
def gelu_table(device):
    with torch.cuda.device(device):
        bits=torch.arange(65536,device=device,dtype=torch.int32).to(torch.int16)
        table=torch.nn.functional.gelu(bits.view(torch.bfloat16),approximate='tanh')
        torch.cuda.current_stream(device).synchronize()
        assert library().set_gelu_table(table.data_ptr())==0
    return table


class RoundedFFN:
    def __init__(self,engine):
        self.enabled=True
        self.original=[]
        for block in engine.wrapper.model.blocks:
            net=block.ffn
            original=net.forward
            self.original.append((net,original))
            linear=net[0]
            gelu_table(linear.weight.device)
            def forward(x,net=net,original=original,linear=linear):
                if (not self.enabled or x.shape[-1]!=1536 or x.numel()!=4608*1536
                        or x.dtype!=torch.bfloat16 or not x.is_contiguous()):
                    return original(x)
                y=x.new_empty((*x.shape[:-1],linear.out_features))
                status=library().ffn_gelu(128,4608,linear.out_features,1536,x.data_ptr(),
                                          linear.weight.data_ptr(),linear.bias.data_ptr(),y.data_ptr(),
                                          torch.cuda.current_stream(x.device).cuda_stream)
                if status:raise RuntimeError(f'CUTLASS FFN returned {status}')
                return net[2](y)
            net.forward=forward

    def close(self):
        for net,original in self.original:net.forward=original
