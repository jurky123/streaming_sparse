"""Immutable baseline, clean timing workers, followed by all eight Val prompts."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from .common import ROOT,PYTHON,config,atomic_json,update,sha
OPTIONS=dict(method='factor',warps=1,pair='tf32x3',tolerance=.001)

ASSIGNMENTS=[('Dense','Dense','old'),('Opt5-K32','K32-Set','old'),('Opt5-V32','V32-Set','old'),
 ('Guarded-K32','K32-Set','new'),('Guarded-V32','V32-Set','new'),('Opt5-K32-replica','K32-Set','old'),
 ('Guarded-K32-replica','K32-Set','new'),('Guarded-V32-replica','V32-Set','new')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root,_=config(a.directory);out=root/'optimization6';out.mkdir(exist_ok=True)
    if a.worker:
        name,mode,variant=ASSIGNMENTS[int(os.environ['RANK'])]
        from . import fast_bench
        if variant=='old':
            from .opt5.network import load_flash
            fast_bench.load_fast=load_flash
        else:
            from .opt6_guarded import load_guarded_head
            fast_bench.load_fast=load_guarded_head
        sys.argv=[sys.argv[0],'--directory',str(root),'--output',str(out/'guard_benchmark'/name),'--mode',mode]
        fast_bench.main();return
    checks=json.loads((out/'checks.json').read_text());assert checks['status']=='completed'and checks['finite_nonempty']
    boundary=json.loads((out/'boundary_checks.json').read_text());assert boundary['status']=='passed'and len(boundary['cases'])==7
    prior=json.loads((root/'optimization6/manifest.json').read_text());old={**prior['code'],**prior['prior_unchanged']}
    for f,d in old.items():assert sha(f)==d,f
    files=[Path(__file__).parent/f for f in ['opt6_guarded.py','opt6_guard_probe.py','opt6_guard_quality.py','opt6_guard_profile.py','opt6_guard_suite.py']]
    manifest=dict(code={str(f.resolve()):sha(f)for f in files},prior_unchanged=old,options=OPTIONS,assignments=ASSIGNMENTS,
        baseline='Opt5 accepted FP16 route; head-only optimization',query_stride=1,global_tf32=False,
        quality_scope='8 registered Val prompts x K32/V32, seed0; Test sealed; frozen weights, no retraining')
    atomic_json(out/'guard_manifest.json',manifest)
    for stage,module,extra in [('guard_benchmark','migration.unified_router.opt6_guard_suite',['--worker']),('guard_quality','migration.unified_router.opt6_guard_quality',[])]:
        update(out/'guard_status.json',status='running',stage=stage,pid=os.getpid());print(stage,flush=True)
        cmd=[str(PYTHON),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus','0,1,2,3,4,5,6,7','--module',module,
             '--logdir',str(out/f'{stage}_logs'),'--','--directory',str(root),*extra]
        with (out/f'{stage}_launcher.log').open('w')as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:update(out/'guard_status.json',status='failed',stage=stage);raise RuntimeError(stage)
    for f,d in {**manifest['code'],**old}.items():assert sha(f)==d,f
    update(out/'guard_status.json',status='completed',stage='benchmark_and_quality_complete',sources_immutable=True)

if __name__=='__main__':main()
