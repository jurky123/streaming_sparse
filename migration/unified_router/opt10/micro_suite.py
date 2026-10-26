import argparse,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);out=root/'optimization10'
for module in ['full_probe','tuned_probe','moment_probe','half_probe','quant_probe','fused_probe']:
 with open(out/(module+'.log'),'w')as f:
  subprocess.run(['streaming_sparse/.venv/bin/python','-u','-m','migration.runtime.bootstrap','migration.unified_router.opt10.'+module,'--directory',str(root)],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),check=True)
 print(module,'completed',flush=True)
