"""Build the optional upstream reference for native_probe.py in /tmp."""
import concurrent.futures
import subprocess
from pathlib import Path
import torch
from local_experiments.video_runtime_v1.weights import ROOT

repo_path='aten/src/ATen/native/transformers/cuda/flash_attn'
work=Path('/tmp/jzk_native_context')
headers=work/'include/ATen/native/transformers/cuda/flash_attn'
headers.mkdir(parents=True,exist_ok=True)
names='alibi.h block_info.h dropout.h flash.h flash_fwd_kernel.h kernel_traits.h mask.h philox.cuh rotary.h softmax.h utils.h'.split()
def fetch(name):
    url=f'https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/{repo_path}/{name}'
    subprocess.run(['curl','-fsSL','--retry','2',url,'-o',str(headers/name)],check=True)
with concurrent.futures.ThreadPoolExecutor(6) as pool:
    list(pool.map(fetch,names))
p=headers/'flash_fwd_kernel.h'
s=p.read_text().replace('bool Return_softmax, typename Params>','bool Return_softmax, bool Compress, typename Params>',1)
s=s.replace('    int n_block = n_block_max - 1;', '    int n_block = Compress ? 1 : n_block_max - 1;',1)
i=s.index('    // These are the iterations where')
s=s[:i]+"""    if constexpr (Compress) {
        #pragma unroll
        for (int i=0;i<size(acc_o);++i) acc_o(i)*=7.f;
        #pragma unroll
        for (int i=0;i<size(softmax.row_sum);++i) softmax.row_sum(i)*=7.f;
    }
"""+s[i:]
p.write_text(s)
torch_dir=Path(torch.__file__).parent
cutlass=ROOT/'external_baselines/Sparse-VideoGen/svg/kernels/3rdparty/flashinfer/3rdparty/cutlass/include'
source=Path(__file__).resolve().parent
flags=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_80','-std=c++17','--expt-relaxed-constexpr',
       '-D_GLIBCXX_USE_CXX11_ABI=0','-I'+str(work/'include'),'-I'+str(torch_dir/'include'),'-I'+str(cutlass),
       '-Xcompiler','-fPIC','--shared','-L'+str(torch_dir/'lib'),'-lc10','-lc10_cuda','-ltorch_cuda','-ltorch_cpu']
for name,args in [('ctx',[]),('ctx_fast',['--use_fast_math'])]:
    subprocess.run(flags+args+[str(source/'ctx.cu'),'-o',str(work/(name+'.so'))],check=True)
for rows,warps in ((32,2),(64,2),(64,4),(128,4)):
    name=f'ctx_{rows}_{warps}.so'
    subprocess.run(flags+[f'-DROWS={rows}',f'-DWARPS={warps}',str(source/'static_ctx.cu'),'-o',str(work/name)],check=True)
