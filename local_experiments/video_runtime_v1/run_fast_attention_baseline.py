"""User-selected fast exact/compensation baseline; paired short-video acceptance."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    common=['--chunks','9','--tile-batch','72','--stream-warps','8','--bf16-qk','--split-summary','--moment-scores','--late-reader','--blocked-scores','--packed-scores','--history-pages','360','--reader-checkpoint','artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt','--local-assets','/tmp/streaming_sparse-0/runtime','--reader-layout','--reader-split']
    for name,mode,components in [('reference','stream','rope'),('current','stream','rope,attention'),('dense','dense','rope')]:
        args=common+['--mode',mode,'--legacy-components',components,'--output',str(OUT/name)]
        with (OUT/(name+'.log')).open('w') as f:subprocess.run([sys.executable,'-u','-m','local_experiments.video_runtime_v1.benchmark_hotspots',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        with (OUT/(name+'_decode.log')).open('w') as f:subprocess.run([sys.executable,'-u','-m','local_experiments.video_runtime_v1.formal_quality','--root',str(OUT/name),'--mode','decode'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
