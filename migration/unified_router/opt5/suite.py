"""Immutable baseline, clean timing workers, followed by all eight Val prompts."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from ..common import ROOT,PYTHON,config,atomic_json,update,sha
from .network import OPTIONS

ASSIGNMENTS=[('Dense','Dense','old'),('Opt4-K32','K32-Set','old'),('Opt4-V32','V32-Set','old'),
 ('Opt5-K32','K32-Set','new'),('Opt5-V32','V32-Set','new'),('Opt4-K32-replica','K32-Set','old'),
 ('Opt5-K32-replica','K32-Set','new'),('Opt5-V32-replica','V32-Set','new')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root,_=config(a.directory);out=root/'optimization5';out.mkdir(exist_ok=True)
    if a.worker:
        name,mode,variant=ASSIGNMENTS[int(os.environ['RANK'])]
        from .. import fast_bench
        if variant=='old':
            from ..opt4_fused import load_fused
            fast_bench.load_fast=lambda *a,**kw:load_fused(*a,method='streamgram',precision='fp16',score_precision='fp16',tile=64,**kw)
        else:
            from .network import load_flash
            fast_bench.load_fast=load_flash
        sys.argv=[sys.argv[0],'--directory',str(root),'--output',str(out/'benchmark'/name),'--mode',mode]
        fast_bench.main();return
    checks=json.loads((out/'checks.json').read_text());assert checks['status']=='completed'and checks['finite_nonempty']
    prior=json.loads((root/'optimization4/fused_manifest.json').read_text());old={**prior['code'],**prior['prior_unchanged']}
    for f,d in old.items():assert sha(f)==d,f
    files=sorted(Path(__file__).parent.glob('*.py'))
    manifest=dict(code={str(f.resolve()):sha(f)for f in files},prior_unchanged=old,options=OPTIONS,assignments=ASSIGNMENTS,
        baseline='Opt4 final fused FP16, explicitly accepted by user',query_stride=1,global_tf32=False,
        quality_scope='8 registered Val prompts x K32/V32, seed0; Test sealed; frozen weights, no retraining')
    atomic_json(out/'manifest.json',manifest)
    for stage,module,extra in [('benchmark','migration.unified_router.opt5.suite',['--worker']),('quality','migration.unified_router.opt5.quality',[])]:
        update(out/'status.json',status='running',stage=stage,pid=os.getpid());print(stage,flush=True)
        cmd=[str(PYTHON),'-u',str(ROOT/'migration/runtime/launch.py'),'--gpus','0,1,2,3,4,5,6,7','--module',module,
             '--logdir',str(out/f'{stage}_logs'),'--','--directory',str(root),*extra]
        with (out/f'{stage}_launcher.log').open('w')as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:update(out/'status.json',status='failed',stage=stage);raise RuntimeError(stage)
    for f,d in {**manifest['code'],**old}.items():assert sha(f)==d,f
    update(out/'status.json',status='completed',stage='benchmark_and_quality_complete',sources_immutable=True)

if __name__=='__main__':main()
