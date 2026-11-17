"""User-selected deployment tiers; shared accepted implementation defaults."""
import argparse
from pathlib import Path
from .deployment import current

TIERS=('base','intersection','step')

def config(tier,**overrides):
    if tier not in TIERS:raise ValueError(tier)
    policy=None if tier=='base' else str(Path(__file__).parent/'presets'/f'{tier}.json')
    return current(full_pset=tier!='base',pset_policy=policy,static_route=True,**overrides)

def main():
    from .benchmark_hotspots import run
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tier',choices=TIERS,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--local-assets')
    p.add_argument('--reps',type=int,default=3)
    p.add_argument('--skip-profile',action='store_true')
    a=p.parse_args()
    run(config(a.tier,output=a.output,local_assets=a.local_assets,reps=a.reps,chunks=9,block_graph=False,skip_profile=a.skip_profile,capture_route=False))

if __name__=='__main__':main()
