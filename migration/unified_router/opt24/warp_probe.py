"""Native CUDA WARP_Q32 variant, retaining same total query tile and masks."""
import argparse,torch,time,json
from pathlib import Path
from .setup import ROOT,tag
from .probe import load,measure
from .attention import attention
from .retile import select
from .sparge import Sparge
from ..opt23.bench import nmse
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',480),('width',832),('chunks',4),('wait_case',4)]:p.add_argument('--'+k,type=int,default=v)
 a=p.parse_args()
 while not (ROOT/'e2e'/f'case{a.wait_case}'/'completed.json').exists():time.sleep(5)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;name=tag(a.height,a.width,a.chunks)
 from torch.utils.cpp_extension import load as load_ext
 build=ROOT/'sparge_warp32_build';build.mkdir(exist_ok=True)
 mod=load_ext(name='opt24_sparge_warp32_sm80',sources=[str(Path(__file__).parent/'sparge_warp32_binding.cu')],extra_include_paths=[str(ROOT/'vendor/sparge/csrc/qattn')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','--expt-relaxed-constexpr','--threads=2'],build_directory=str(build),verbose=True)
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,24);q=aux['q_packed'];kv=aux['kv'];ix=aux['q_order'];b=kv.state['key'].shape[1]
 base=lambda:attention(q,kv,aux['selection'],pipe.quant,pipe.seg,ix);bg,bo=capture(base);rows=[]
 for group in [32,64,128]:
  ids=select(aux['fine'],aux['candidates'],b,group);lib=Sparge(kv,pipe.seg,group);prior=lib.prepared(q,ids,ix);lib.module=mod
  y=lib.prepared(q,ids,ix);err=nmse(y,prior);assert err<1e-4,(group,err)
  timing(bg,300 if q.shape[1]<10000 else 15);r,y=measure(lambda:lib.prepared(q,ids,ix),bg,ref,reps=20);r.update(group=group,warp_q=32,nmse_vs_warp16=err,speedup=r['baseline_ms']/r['ms']);rows.append(r);print(r,flush=True)
 atomic_json(ROOT/f'warp32_{name}.json',dict(completed=True,rows=rows))
if __name__=='__main__':main()
