"""Frozen clean speed comparison, followed by quality and profiling."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from ..common import ROOT,PYTHON,config,atomic_json,update,sha

ASSIGNMENTS=[('Dense','Dense',False),('Fast1-K32','K32-Set',False),('Fast1-V32','V32-Set',False),
    ('Opt2-K32','K32-Set',True),('Opt2-V32','V32-Set',True),('Opt2-K32-replica','K32-Set',True),
    ('Opt2-V32-replica','V32-Set',True),('Opt2-U-Sol','U-Sol',True)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root,_=config(a.directory);out=root/'optimization2';out.mkdir(exist_ok=True)
    if a.worker:
        rank=int(os.environ['RANK']);name,mode,new=ASSIGNMENTS[rank]
        from .. import fast_bench
        if new:
            from .network import load_optimized
            fast_bench.load_fast=load_optimized
        sys.argv=[sys.argv[0],'--directory',str(root),'--output',str(out/'benchmark'/name),'--mode',mode,
            '--graph','--fused','--history-cache','--compiled-build']
        fast_bench.main();return
    prior=json.loads((root/'optimization/final_manifest.json').read_text())
    for f,d in {**prior['code'],**prior['unmodified_original']}.items():assert sha(f)==d,f
    checks=json.loads((out/'correctness.json').read_text());assert checks['finite_and_nonempty']
    files=list(Path(__file__).parent.glob('*.py'))
    manifest=dict(code={str(f.resolve()):sha(f)for f in files},prior_code=prior['code'],original_code=prior['unmodified_original'],
        query_stride=1,attention_schedule=dict(query_block=32,warps=4,stages=3,individual_masks=True),
        action_head_factorization=False,streaming_moments=False,tf32=False,
        assignments=ASSIGNMENTS,quality_scope='All8 registered Val prompts, seed0, checkpoints frozen; Test untouched')
    atomic_json(out/'manifest.json',manifest)
    for stage,module,extra in [('benchmark','migration.unified_router.opt2.suite',['--worker']),('quality','migration.unified_router.opt2.quality',[])]:
        update(out/'status.json',status='running',stage=stage,pid=os.getpid());print(stage,flush=True)
        cmd=[str(PYTHON),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus','0,1,2,3,4,5,6,7','--module',module,
            '--logdir',str(out/f'{stage}_logs'),'--','--directory',str(root),*extra]
        with (out/f'{stage}_launcher.log').open('w')as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:update(out/'status.json',status='failed',stage=stage);raise RuntimeError(stage)
    for f,d in {**manifest['code'],**manifest['prior_code'],**manifest['original_code']}.items():assert sha(f)==d,f
    update(out/'status.json',status='completed',stage='benchmark_and_quality_complete',sources_immutable=True)

if __name__=='__main__':main()
