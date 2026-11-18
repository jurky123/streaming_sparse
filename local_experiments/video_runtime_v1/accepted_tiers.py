"""User-selected deployment tiers; shared accepted implementation defaults."""
import argparse
from pathlib import Path
from .deployment import CURRENT,current

TIERS=('base','intersection','step')

def config(tier,**overrides):
    if tier not in TIERS:raise ValueError(tier)
    policy=None if tier=='base' else str(Path(__file__).parent/'presets'/f'{tier}.json')
    if overrides.get('anchor_scale') is None:
        overrides['anchor_scale']=tier=='base'
    if overrides.get('column_mass') is None:
        overrides['column_mass']=tier=='base' and overrides['anchor_scale']
    return current(full_pset=tier!='base',pset_policy=policy,static_route=True,**overrides)

def main():
    from .benchmark_hotspots import run
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--column-mass',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--anchor-scale',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--repeated-context',action=argparse.BooleanOptionalAction,default=CURRENT['repeated_context'])
    p.add_argument('--radix-select',action=argparse.BooleanOptionalAction,default=CURRENT['radix_select'])
    p.add_argument('--channel-major-stats',action=argparse.BooleanOptionalAction,default=CURRENT['channel_major_stats'])
    p.add_argument('--summary-window',action=argparse.BooleanOptionalAction,default=CURRENT['summary_window'])
    p.add_argument('--residual-norm',action=argparse.BooleanOptionalAction,default=CURRENT['residual_norm'])
    p.add_argument('--resident-route',action=argparse.BooleanOptionalAction,default=CURRENT['resident_route'])
    p.add_argument('--cached-projection',action=argparse.BooleanOptionalAction,default=CURRENT['cached_projection'])
    p.add_argument('--fused-exact-tail',action=argparse.BooleanOptionalAction,default=CURRENT['fused_exact_tail'])
    p.add_argument('--scale-in-qk', action=argparse.BooleanOptionalAction, default=CURRENT['scale_in_qk'])
    p.add_argument('--native-route-qk', action=argparse.BooleanOptionalAction, default=CURRENT['native_route_qk']);p.add_argument('--rms-row',action=argparse.BooleanOptionalAction,default=CURRENT['rms_row']);p.add_argument('--tier',choices=TIERS,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--local-assets')
    p.add_argument('--reps',type=int,default=3)
    p.add_argument('--grouping-softmax',action='store_true')
    p.add_argument('--grouping-spatial',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--grouping-dataflow',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--risk-online',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--summary-moments',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--native-rope',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--exact-fold',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--qk-tile64',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--rms-mean',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--rms-dataflow',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--block-pointwise',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--block-graph',action='store_true')
    p.add_argument('--skip-profile',action='store_true')
    a=p.parse_args()
    run(config(a.tier,column_mass=a.column_mass,anchor_scale=a.anchor_scale,repeated_context=a.repeated_context,radix_select=a.radix_select,channel_major_stats=a.channel_major_stats,residual_norm=a.residual_norm,resident_route=a.resident_route,summary_window=a.summary_window,cached_projection=a.cached_projection,fused_exact_tail=a.fused_exact_tail,scale_in_qk=a.scale_in_qk and a.native_route_qk,native_route_qk=a.native_route_qk,rms_row=a.rms_row,risk_online=a.risk_online,summary_moments=a.summary_moments,native_rope=a.native_rope,exact_fold=a.exact_fold,qk_tile64=a.qk_tile64,rms_mean=a.rms_mean,rms_dataflow=a.rms_dataflow,block_pointwise=a.block_pointwise,grouping_spatial=a.grouping_spatial,grouping_softmax=a.grouping_softmax,grouping_dataflow=a.grouping_dataflow,output=a.output,local_assets=a.local_assets,reps=a.reps,chunks=9,block_graph=a.block_graph,skip_profile=a.skip_profile,capture_route=False))

if __name__=='__main__':main()
