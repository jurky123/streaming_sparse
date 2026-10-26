"""One GPU per geometry, sequential layers; fail visibly instead of retry loops."""
import argparse,subprocess,sys,os,json
from .setup import CASES,ROOT,tag
from ..common import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);a=p.parse_args();h,w,c=CASES[a.case];name=tag(h,w,c);folder=ROOT/'bench'/name;folder.mkdir(parents=True,exist_ok=True)
 for layer in [24,1,14,29]:
  if (folder/f'layer{layer:02d}.json').exists():continue
  with (folder/f'layer{layer:02d}.log').open('w') as log:
   r=subprocess.run([sys.executable,'-u','-m','migration.runtime.bootstrap','migration.unified_router.opt23.bench','--height',str(h),'--width',str(w),'--chunks',str(c),'--layer',str(layer)],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:
   atomic_json(folder/'worker.json',dict(status='failed',layer=layer,code=r.returncode));raise SystemExit(r.returncode)
 atomic_json(folder/'worker.json',dict(status='completed'))
if __name__=='__main__':main()
