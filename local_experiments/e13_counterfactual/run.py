import sys,subprocess,concurrent.futures
from .core import *
def launch(stage,args=()):
 (OUT/'logs').mkdir(parents=True,exist_ok=True);tag=stage+'_'+'_'.join(args)
 with (OUT/'logs'/(tag+'.log')).open('a') as f:subprocess.run([sys.executable,'-m','local_experiments.e13_counterfactual.'+stage,*args],stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
if __name__=='__main__':
 preregister();assert json.loads((OUT/'protocol.json').read_text())['manifest_sha256']==sha(DATA/'manifest.json')
 if not (OUT/'preparation.json').exists():launch('prepare')
 configs=[('A0','Risk'),('A2','Risk'),('A0','CF'),('A2','CF')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  jobs=[ex.submit(launch,'train',('--arch',a,'--regime',r,'--gpu',str(i))) for i,(a,r) in enumerate(configs)]
  for j in jobs:j.result()
 for stage,marker in [('evaluate','evaluation_summary.json'),('mechanism','mechanism_summary.json'),('diagnostics','conditional_rank_summary.json'),('audit','audit.json')]:
  if not (OUT/marker).exists():launch(stage)
 launch('report')
