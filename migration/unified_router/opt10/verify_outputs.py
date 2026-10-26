"""Verify all timed outputs, including controls, against independent quality runs."""
import argparse,json,hashlib
from pathlib import Path
import torch
from ..common import atomic_json
p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory);out=root/'optimization10';pids=json.loads((root/'e2e_manifest.json').read_text())['prompt_ids'];checked=0
for i,pid in enumerate(pids):
 expected={}
 for name in ['Dense','Opt9','Wide','Head','Moments','Compact']:
  if name=='Dense':latent=torch.load(root/f'videos/{pid}.seed0/latents.pt',map_location='cpu',weights_only=True)['Dense']
  elif name=='Opt9':latent=torch.load(root/f'optimization9/quality/{pid}.seed0/MemoryFull.latent.pt',map_location='cpu',weights_only=True)
  else:latent=torch.load(out/f'quality/{pid}.seed0/{name}.latent.pt',map_location='cpu',weights_only=True)
  expected[name]=hashlib.sha256(latent.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 assert expected['Wide']==expected['Opt9']
 assert expected['Compact']==expected['Moments']
 folder=out/('paired'if i==0 else f'paired_val/prompt{i}');data=json.loads((folder/'timings.json').read_text());assert len(data['records'])==(data['repeats']+1)*6
 for row in data['records']:
  assert row['latent_sha256']==expected[row['variant']],(pid,row['variant'],row['round']);checked+=1
atomic_json(out/'checksum_verification.json',dict(timed_outputs_checked=checked,all_match_independent_quality=True,wide_bit_identical_all8=True,compact_bit_identical_to_moments_all8=True));print('checked',checked,flush=True)
