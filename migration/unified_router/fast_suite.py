"""Controlled clean benchmarks first, then fixed Val regression on eight GPUs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .common import ROOT,PYTHON,config,atomic_json,update,sha

ASSIGNMENTS=[('Dense','Dense',1,16),('U-Sol','U-Sol',1,16),('Fast-K32','K32-Set',1,16),
    ('Fast-V32','V32-Set',1,16),('Q16-K32','K32-Set',16,16),('Q16-V32','V32-Set',16,16),
    ('Q64-K32','K32-Set',64,64),('Fast-K32-replica','K32-Set',1,16)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root,_=config(a.directory);out=root/'optimization';out.mkdir(exist_ok=True)
    if a.worker:
        rank=int(os.environ['RANK']);name,mode,stride,group=ASSIGNMENTS[rank]
        from .fast_bench import main as bench
        sys.argv=[sys.argv[0],'--directory',str(root),'--output',str(out/'final_benchmark'/name),
            '--mode',mode,'--graph','--fused','--history-cache','--compiled-build','--query-stride',str(stride),'--group',str(group)]
        bench();return
    # Earlier ablations finish before allocating their devices to this suite.
    for name in ['stage4_q16_K32','stage4_q64_K32','stage3_fused_K32']:
        while True:
            path=out/name/'status.json'
            if path.exists()and json.loads(path.read_text())['status']=='completed':break
            update(out/'status.json',status='waiting',stage='finishing_ablations',waiting_for=name)
            time.sleep(5)
    files=sorted((ROOT/'migration/unified_router').glob('fast_*.py'))
    manifest=dict(code={str(f):sha(f)for f in files},benchmark_assignments=ASSIGNMENTS,
        comparison='Same first registered Val prompt and generation seed0, single GPU, 1 warmup + 3 timed repeats',
        regression='All eight registered Val prompts, generation seed0, no Test access',
        unmodified_original=json.loads((root/'e2e_manifest.json').read_text())['code_sha256'])
    atomic_json(out/'final_manifest.json',manifest)
    for stage,module,extra in [('clean_benchmark','migration.unified_router.fast_suite',['--worker']),
                                ('quality_regression','migration.unified_router.fast_quality',['--output',str(out/'quality_seed0')])]:
        update(out/'status.json',status='running',stage=stage,pid=os.getpid())
        cmd=[str(PYTHON),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus','0,1,2,3,4,5,6,7',
            '--module',module,'--logdir',str(out/f'{stage}_logs'),'--','--directory',str(root),*extra]
        print('Running',stage,flush=True)
        with (out/f'{stage}_launcher.log').open('w')as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            update(out/'status.json',status='failed',stage=stage);raise RuntimeError(stage)
    for f,digest in {**manifest['code'],**manifest['unmodified_original']}.items():assert sha(f)==digest,f
    update(out/'status.json',status='completed',stage='benchmarks_and_quality_complete',immutable_source_verified=True)

if __name__=='__main__':main()
