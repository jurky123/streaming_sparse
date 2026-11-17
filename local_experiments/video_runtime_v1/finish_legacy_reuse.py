"""Same-GPU speed control and decoding for the legacy-reuse short-video test."""
import argparse,subprocess,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/video_runtime_v1/legacy_reuse'

def call(module,args,logname):
    with (OUT/logname).open('w') as f:subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',action='store_true');p.add_argument('--decode');a=p.parse_args()
    bench='local_experiments.video_runtime_v1.benchmark_hotspots';quality='local_experiments.video_runtime_v1.formal_quality'
    common=['--chunks','9','--history-pages','360','--local-assets','/tmp/streaming_sparse-0/runtime','--reader-checkpoint','artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt','--skip-profile']
    if a.control:
        name='baseline_samegpu'
        call(bench,['--mode','stream',*common,'--tile-batch','72','--stream-warps','8','--bf16-qk','--split-summary','--moment-scores','--late-reader','--blocked-scores','--packed-scores','--output',str(OUT/name)],name+'.log')
        for name in ['all','baseline_samegpu']:call(quality,['--mode','decode','--root',str(OUT/name)],'decode_'+name+'_samegpu.log')
        call(bench,['--mode','dense_native',*common,'--output',str(OUT/'dense_samegpu')],'dense_samegpu.log')
        call(quality,['--mode','decode','--root',str(OUT/'dense_samegpu')],'decode_dense_samegpu.log')
    else:call(quality,['--mode','decode','--root',str(OUT/a.decode)],'decode_'+a.decode+'.log')
if __name__=='__main__':main()
