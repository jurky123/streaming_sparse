"""Run all six candidates on the restored and fingerprinted SAME raw Q/K/V."""
import argparse,json,torch,hashlib
from pathlib import Path
from ..review_v1.model import Model
from ..learned_cache.io import atomic
from .run import ROOT,BASE,PRE
from .local import evaluate
from .core import GROUPS
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();key=f'{a.case}_s{a.seed}'
 assert json.loads((ROOT/'restoration'/f'{key}.json').read_text())['latent_bitwise_equal']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
 for p in sorted((ROOT/'qkv'/key).glob('*.pt')):
  dest=ROOT/'local'/key/p.with_suffix('.json').name
  if dest.exists():continue
  meta=json.loads((ROOT/'restored_controls'/key/dest.name).read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['qkv_sha256']
  x=torch.load(p,map_location='cuda',weights_only=True);identity={k:meta[k] for k in ['case','seed','chunk','layer','step','history_pages','qkv_path','qkv_sha256']};identity['source']='restored original same-QKV event; full baseline latent bitwise verified'
  rows=evaluate(model[meta['layer']],x['q'],x['k'],x['v'],x['ids'],x['tiles'],dest,identity)
  old=json.loads((Path('/tmp/group_response_results/local')/key/dest.name).read_text())
  for g in ['G-ORDER','G-K','G-KVR']:
   for panel in ['representation','mass','risk']:
    for metric in ['nmse','squared_error','target_energy','log_denominator_mae']:
     aa=rows[g][panel][metric];bb=old['rows'][g][panel][metric];assert abs(aa-bb)<=1e-6*max(abs(bb),1e-8),(g,panel,metric,aa,bb)
  atomic(dest.with_suffix('.reuse'),dict(passed=True,old_three_groups_match=True));print('LOCAL',key,p.name,flush=True)
if __name__=='__main__':main()
