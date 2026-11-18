"""Square inside the fixed PyTorch D1536 mean tree; no FP32 square array."""
import ctypes,hashlib,subprocess
from functools import lru_cache
from pathlib import Path
from filelock import FileLock
import torch,triton
from .rms_dataflow import _finish
@lru_cache(None)
def library():
 source=Path(__file__).parent/'csrc/rms_mean.cu';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
 folder=Path('/tmp/streaming_sparse-0/rms_mean')/digest;folder.mkdir(parents=True,exist_ok=True);binary=folder/'kernel.so'
 with FileLock(str(folder/'build.lock')):
  if not binary.exists():subprocess.run(['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_80','--fmad=false','--shared','-Xcompiler','-fPIC',str(source),'-o',str(binary)],check=True)
 lib=ctypes.CDLL(str(binary));lib.rms_mean.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];lib.rms_mean.restype=None
 lib.rms_row.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
 lib.rms_row.restype=None
 lib.rms_rope.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_float,ctypes.c_void_p]+[ctypes.c_int]*3+[ctypes.c_void_p]
 lib.rms_rope.restype=None
 return lib

def mean_square(x):
 assert x.is_cuda and x.is_contiguous() and x.dtype==torch.bfloat16 and x.shape[-1]==1536 and x.numel()//1536>=16
 mean=torch.empty((*x.shape[:-1],1),device=x.device,dtype=torch.float32)
 library().rms_mean(x.data_ptr(),mean.data_ptr(),x.numel()//1536,torch.cuda.current_stream().cuda_stream)
 return mean

def fused(x,weight,eps):
 inv=torch.rsqrt(mean_square(x)+eps);out=torch.empty_like(x)
 _finish[(triton.cdiv(x.numel(),1024),)](x,inv,weight,out,x.numel(),1536,1024,num_warps=4,enable_fp_fusion=False)
 return out

def row_fused(x, weight, eps):
 if not (x.is_contiguous() and weight.is_contiguous() and x.dtype == weight.dtype == torch.bfloat16
         and x.shape[-1] == weight.numel() == 1536 and x.numel() // 1536 >= 16
         and x.data_ptr() % 8 == weight.data_ptr() % 8 == 0):
  return fused(x, weight, eps)
 out = torch.empty_like(x)
 library().rms_row(x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.numel() // 1536,
                   eps, torch.cuda.current_stream(x.device).cuda_stream)
 return out

def row_rope(x,weight,eps,freq,gh,gw,start,out=None):
 assert x.shape==(1,4608,1536) and x.is_contiguous() and weight.is_contiguous()
 assert x.dtype==weight.dtype==torch.bfloat16 and freq.is_contiguous() and freq.dtype==torch.complex128
 if out is None:out=torch.empty_like(x)
 else:assert out.is_contiguous() and out.shape==(1,4608,12,128) and out.dtype==x.dtype and out.device==x.device
 library().rms_rope(x.data_ptr(),weight.data_ptr(),out.data_ptr(),4608,eps,
                    freq.data_ptr(),gh,gw,start,torch.cuda.current_stream(x.device).cuda_stream)
 return out.view(1,4608,12,128)

def project_rotary(module,x,grid,freq,start,key_out=None):
 frames,gh,gw=grid[0].tolist()
 assert frames*gh*gw==x.shape[1] and 0<=start<=freq.shape[0]-frames
 q=row_rope(module.q(x),module.norm_q.weight,module.norm_q.eps,freq,gh,gw,start)
 k=row_rope(module.k(x),module.norm_k.weight,module.norm_k.eps,freq,gh,gw,start,key_out)
 return q,k


def project_cached(module,x,grid,freq,start,key_out,value_out):
 q,k=project_rotary(module,x,grid,freq,start,key_out)
 width=x.shape[-1]
 torch.addmm(module.v.bias,x.reshape(-1,width),module.v.weight.T,out=value_out.view(-1,width))
 return q,k,value_out


def set_rotary(engine,enabled):
 for block in engine.wrapper.model.blocks:
  block.self_attn.project_normalized_rotary=project_rotary if enabled else None

def set_cached_projection(engine,enabled):
 for block in engine.wrapper.model.blocks:
  block.self_attn.project_cached_rotary=project_cached if enabled else None

class NormalizedRotary:
 def __init__(self,engine,cached_projection=False):
  self.engine=engine
  set_rotary(engine,True)
  set_cached_projection(engine,cached_projection)

 def close(self):
  set_rotary(self.engine,False)
  set_cached_projection(self.engine,False)
