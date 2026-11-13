"""Verify and restore published assets to their original experiment paths."""
import argparse,gzip,hashlib,json,os,shutil,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'reproduction/weights_20261113'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def restore(destination,verify_only=False):
 destination=Path(destination).resolve()
 manifest=json.loads((BUNDLE/'manifest.json').read_text())
 for row in manifest['assets']:
  payload=BUNDLE/row['payload'];target=(destination/row['path']).resolve()
  if destination not in target.parents:raise ValueError('Asset path escapes destination')
  if sha(payload)!=row['compressed_sha256']:raise ValueError(f'Corrupt payload: {payload}')
  if target.exists():
   if sha(target)!=row['sha256']:raise ValueError(f'Existing file differs; preserve/move it before restoring: {target}')
  elif verify_only:raise FileNotFoundError(target)
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   fd,tmp=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=target.parent)
   try:
    with os.fdopen(fd,'wb') as out,gzip.open(payload,'rb') as inp:shutil.copyfileobj(inp,out)
    if Path(tmp).stat().st_size!=row['bytes'] or sha(Path(tmp))!=row['sha256']:raise ValueError(f'Uncompressed checksum mismatch: {target}')
    os.replace(tmp,target)
   finally:
    if os.path.exists(tmp):os.unlink(tmp)
  print('OK',row['name'],flush=True)
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=ROOT);p.add_argument('--verify-only',action='store_true');a=p.parse_args();restore(a.destination,a.verify_only)
