"""Rebuild the independent experiment import tree without resetting the user repo."""
import argparse,subprocess,tarfile,io,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime',default='/tmp/streaming_sparse-0/runtime');p.add_argument('--destination',default='/tmp/group_response_178f95a8');a=p.parse_args()
src=Path(__file__).resolve().parents[1];fast=Path(a.runtime);dest=Path(a.destination);assert (fast/'streaming_sparse/.venv/bin/python').exists();dest.mkdir(parents=True,exist_ok=True)
blob=subprocess.check_output(['git','archive','178f95a8','local_experiments'],cwd=src)
with tarfile.open(fileobj=io.BytesIO(blob)) as t:t.extractall(dest)
for f in fast.iterdir():
 if f.name!='local_experiments' and not (dest/f.name).exists():(dest/f.name).symlink_to(f,target_is_directory=f.is_dir())
shutil.copytree(src/'local_experiments/group_response',dest/'local_experiments/group_response',dirs_exist_ok=True)
print('Prepared',dest,'at commit 178f95a8; run only from this directory to avoid importing modified workspace baseline modules.')
