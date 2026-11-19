"""Prepare unchanged 178f95a8 baseline modules and both independent experiment packages."""
import argparse,subprocess,io,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime',default='/tmp/streaming_sparse-0/runtime');p.add_argument('--destination',default='/tmp/group_response_178f95a8');a=p.parse_args();src=Path(__file__).resolve().parents[1];dest=Path(a.destination);fast=Path(a.runtime);dest.mkdir(parents=True,exist_ok=True)
assert (fast/'streaming_sparse/.venv/bin/python').exists()
blob=subprocess.check_output(['git','archive','178f95a8','local_experiments'],cwd=src)
with tarfile.open(fileobj=io.BytesIO(blob)) as t:t.extractall(dest)
for f in fast.iterdir():
 if f.name!='local_experiments' and not (dest/f.name).exists():(dest/f.name).symlink_to(f,target_is_directory=f.is_dir())
for name in ['group_response','group_hierarchy']:shutil.copytree(src/'local_experiments'/name,dest/'local_experiments'/name,dirs_exist_ok=True)
print(dest)
