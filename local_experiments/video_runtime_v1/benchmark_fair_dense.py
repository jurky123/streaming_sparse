"""Native Dense and retained sparse tiers with explicitly shared backbone work."""
import argparse
from .deployment import CURRENT,current
from .accepted_tiers import config
from .benchmark_hotspots import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['dense_original','dense_recent','dense','base','intersection','step'],required=True);p.add_argument('--output',required=True);p.add_argument('--reps',type=int,default=5);p.add_argument('--native-rope',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args()
 common=dict(output=a.output,reps=a.reps,chunks=9,block_graph=False,skip_profile=True,capture_route=False,local_assets='/tmp/streaming_sparse-0/runtime')
 if a.method.startswith('dense'):
  enabled=a.method!='dense_original'
  cfg=current(mode='dense_native',repeated_context=a.method=='dense' and CURRENT['repeated_context'],rms_rope=a.method=='dense' and CURRENT['rms_rope'],layernorm_mod=a.method=='dense' and CURRENT['layernorm_mod'],rounded_ffn=a.method=='dense' and CURRENT['rounded_ffn'],full_pset=False,static_route=True,block_pointwise=enabled,rms_dataflow=enabled,rms_mean=enabled,native_rope=a.method=='dense',**common)
 else:cfg=config(a.method,native_rope=a.native_rope,**common)
 run(cfg)
if __name__=='__main__':main()
