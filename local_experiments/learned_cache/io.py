import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import torch


def atomic(path,value,tensor=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb' if tensor else 'w') as stream:
        if tensor:torch.save(value,stream)
        else:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def disk_guard(root,min_free_gib=50,max_run_gib=2):
    free=shutil.disk_usage(root).free
    used=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())
    if free<min_free_gib*2**30 or used>max_run_gib*2**30:
        raise RuntimeError(f'Disk guard: free={free/2**30:.2f} GiB, run={used/2**30:.3f} GiB')
    return dict(free_gib=free/2**30,run_mib=used/2**20)


class RunLock:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.fd=(self.root/'run.lock').open('a+')
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Another worker holds {self.root}/run.lock')
        self.fd.seek(0);self.fd.truncate();self.fd.write(json.dumps(dict(pid=os.getpid(),host=socket.gethostname())));self.fd.flush()
    def close(self):
        fcntl.flock(self.fd,fcntl.LOCK_UN);self.fd.close()


def checkpoint(root,state):
    root=Path(root)
    # Serialize first. Keep the previous completed checkpoint even if killed
    # during writing, renaming or fsync. No growing per-step checkpoint list.
    atomic(root/'next.pt',state,True)
    if (root/'latest.pt').exists():os.replace(root/'latest.pt',root/'previous.pt')
    os.replace(root/'next.pt',root/'latest.pt')
    fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)


def restore(root):
    errors=[]
    for name in ['latest.pt','next.pt','previous.pt']:
        path=Path(root)/name
        if path.exists():
            try:return torch.load(path,map_location='cpu',weights_only=False),name
            except Exception as e:errors.append(f'{name}: {e}')
    if errors:raise RuntimeError('No valid checkpoint: '+'; '.join(errors))
    return None,None
