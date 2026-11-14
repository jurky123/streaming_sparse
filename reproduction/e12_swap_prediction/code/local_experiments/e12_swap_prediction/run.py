"""Reproduce E12 with predeclared gates; reuse completed immutable results."""
import os,sys,json,subprocess,concurrent.futures,argparse
from .core import *
def launch(stage,args=()):
 logdir=OUT/'logs';logdir.mkdir(parents=True,exist_ok=True);name=stage+'_'+('_'.join(str(s).replace('/','_') for s in args) or 'main')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (logdir/(name+'.log')).open('a') as f:subprocess.run([sys.executable,'-m','local_experiments.e12_swap_prediction.'+stage,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def run(gpus):
 preregister();assert json.loads((OUT/'protocol.json').read_text())['manifest_sha256']==sha(DATA/'manifest.json')
 if not (OUT/'preparation.json').exists():launch('prepare')
 with concurrent.futures.ThreadPoolExecutor(max_workers=min(3,len(gpus))) as ex:
  jobs=[ex.submit(launch,'train',('--arch',arch,'--gpu',str(gpus[i%len(gpus)]))) for i,arch in enumerate(('A0','A1','A2'))]
  for job in jobs:job.result()
 primary=[json.loads((OUT/'models'/f'{a}_s17'/'selection.json').read_text()) for a in ('A0','A1','A2')];passed=[x for x in primary if x['selected']['gate']];winner=max(passed,key=lambda x:x['selected']['mean_gain_pct'])['architecture'] if passed else None
 if not (OUT/'development_gate.json').exists():dump(OUT/'development_gate.json',dict(primary=primary,winner=winner,second_seed=29 if winner else None,evaluation_accessed=False))
 if winner:launch('train',('--arch',winner,'--seed','29','--gpu',str(gpus[0])))
 if not (OUT/'evaluation_summary.json').exists():launch('evaluate')
 if not (OUT/'audit.json').exists():launch('audit')
 if not (OUT/'reference_cost.json').exists():launch('benchmark')
 launch('report')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--gpus',default='0,1,2');args=parser.parse_args();run([int(x) for x in args.gpus.split(',')])
