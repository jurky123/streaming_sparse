"""Same-device cached R4 log-mass short video acceptance."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/logmass_video_v1'

def main():
    common=['--mode','stream','--chunks','9','--tile-batch','72','--stream-warps','8','--bf16-qk','--split-summary','--moment-scores','--late-reader','--blocked-scores','--packed-scores','--history-pages','360','--reader-checkpoint','artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt','--local-assets','/tmp/streaming_sparse-0/runtime','--reader-layout','--reader-split','--scan-exclude','--legacy-components','rope,attention','--projected-risk']
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['baseline','cached']:
        args=common+['--output',str(OUT/name)]+(['--cached-logmass'] if name=='cached' else [])
        with (OUT/(name+'.log')).open('w') as f:subprocess.run([sys.executable,'-u','-m','local_experiments.video_runtime_v1.benchmark_hotspots',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        with (OUT/(name+'_decode.log')).open('w') as f:subprocess.run([sys.executable,'-u','-m','local_experiments.video_runtime_v1.formal_quality','--root',str(OUT/name),'--mode','decode'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
