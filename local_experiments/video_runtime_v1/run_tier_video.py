"""Warm paired short-video tiers; implementation and inference policy are separate axes."""
import argparse,subprocess,sys
from .weights import ROOT
from .deployment import current

def main():
 p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['full_q64','fast_q64']);p.add_argument('--child');a=p.parse_args()
 root=ROOT/'artifacts/video_runtime_v1/tiers_m16_v1';root.mkdir(parents=True,exist_ok=True)
 if a.child:
  from .benchmark_hotspots import run
  name=a.child
  args=current(full_pset=not name.startswith('fast'),pset_steps='last' if name.startswith('last') else 'all',route_m16=name.endswith('m16'),reader_vector_g2='g2' in name,local_assets='/tmp/streaming_sparse-0/runtime',output=str(root/name),chunks=9,reps=3,block_graph=False,skip_profile=False,capture_route=False)
  run(args);return
 for name in a.names:
  with (root/(name+'.log')).open('w') as f:subprocess.run([sys.executable,'-u','-m',__package__+'.run_tier_video','--child',name],stdout=f,stderr=subprocess.STDOUT,check=True)
  with (root/(name+'_decode.log')).open('w') as f:subprocess.run([sys.executable,'-u','-m',__package__+'.formal_quality','--root',str(root/name),'--mode','decode'],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
