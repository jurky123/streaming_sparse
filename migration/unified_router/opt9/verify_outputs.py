"""Verify every timed checksum, including Dense and prior-version controls."""
import argparse,json,hashlib
from pathlib import Path
import torch
from ..common import atomic_json
p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);out=root/'optimization9';pids=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids'];checked=0
for i,pid in enumerate(pids):
 expected={}
 for name in ['Dense','Opt8Full','Opt8Fast','Packed','Skip2','Skip0','Skip2Fast','Skip0Fast','MemoryFull','MemoryFast']:
  if name=='Dense':latent=torch.load(root/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)['Dense']
  elif name in ['Opt8Full','Opt8Fast']:
   old='QKProj'if name=='Opt8Full'else'QKFast';latent=torch.load(root/f'optimization8/quality/{pid}.seed0/{old}.latent.pt',map_location='cpu',weights_only=True)
  else:latent=torch.load(out/f'quality/{pid}.seed0/{name}.latent.pt',map_location='cpu',weights_only=True)
  expected[name]=hashlib.sha256(latent.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 assert expected['MemoryFull']==expected['Packed'];assert expected['MemoryFast']==expected['Opt8Fast']
 for prefix in ['paired','memory_paired']:
  folder=out/(prefix if i==0 else f'{prefix}_val/prompt{i}');data=json.loads((folder/'timings.json').read_text())
  assert len(data['records'])==(data['repeats']+1)*(8 if prefix=='paired'else 6)
  for row in data['records']:
   assert row['latent_sha256']==expected[row['variant']],(pid,prefix,row['variant'],row['round']);checked+=1
atomic_json(out/'checksum_verification.json',dict(timed_outputs_checked=checked,all_match_quality_including_dense_and_opt8=True,memory_rewrite_bit_identical_all16=True));print('checked',checked,flush=True)
