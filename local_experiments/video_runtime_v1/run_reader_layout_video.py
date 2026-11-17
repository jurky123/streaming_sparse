"""Paired short-video acceptance for the exact A-only Reader layout."""
import subprocess,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/video_runtime_v1/reader_layout_v1'

def call(module,args,log):
    with log.open('w') as f:subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy',action='store_true');p.add_argument('--rope',action='store_true');a=p.parse_args()
    common=['--mode','stream','--chunks','9','--tile-batch','72','--stream-warps','8','--bf16-qk','--split-summary','--moment-scores','--late-reader','--blocked-scores','--packed-scores','--history-pages','360','--reader-checkpoint','artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt','--local-assets','/tmp/streaming_sparse-0/runtime']
    variants=['legacy_layout'] if a.legacy else ['rope_layout'] if a.rope else ['baseline','layout']
    for name in variants:
        args=common+['--output',str(OUT/name)]
        if name!='baseline':args+=['--reader-layout']
        if a.legacy:args+=['--legacy-components','rope,risk,attention,summary']
        elif a.rope:args+=['--legacy-components','rope']
        if not (OUT/name/'result.json').exists():call('local_experiments.video_runtime_v1.benchmark_hotspots',args,OUT/(name+'.log'))
        call('local_experiments.video_runtime_v1.formal_quality',['--root',str(OUT/name),'--mode','decode'],OUT/(name+'_decode.log'))
if __name__=='__main__':main()
