"""Restore the pinned evaluator sources and verified public evaluation weights."""
import json
import subprocess
import zipfile
from paths import CODE,RUN
from prepare import sha

def checkout(path,url,revision):
    if not path.exists():
        subprocess.run(['git','clone',url,path],check=True)
        subprocess.run(['git','-C',path,'checkout','--detach',revision],check=True)
    if subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()!=revision:
        raise ValueError(f'unexpected revision: {path}')

if __name__=='__main__':
    vbench=RUN/'VBench-master'
    checkout(vbench,'https://github.com/Vchitect/VBench.git','fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490')
    cmd=['git','-C',vbench,'apply']
    patch=CODE/'vbench.patch'
    if subprocess.run([*cmd,'--reverse','--check',patch],capture_output=True).returncode:
        subprocess.run([*cmd,patch],check=True)
    checkout(RUN/'cache/dino_model/facebookresearch_dino_main','https://github.com/facebookresearch/dino.git','7c446df5b9f45747937fb0d72314eb9f7b66930a')
    for name,spec in json.loads((CODE/'score-assets.json').read_text()).items():
        p=RUN/name
        if not p.exists():
            p.parent.mkdir(parents=True,exist_ok=True)
            tmp=p.with_name(p.name+'.part')
            subprocess.run(['curl','-fL','--retry','3','-o',tmp,spec['url']],check=True)
            if sha(tmp)!=spec['sha256']:raise ValueError(f'download hash mismatch: {name}')
            tmp.replace(p)
        elif sha(p)!=spec['sha256']:raise ValueError(f'hash mismatch: {name}')
    raft=RUN/'cache/raft_model'
    with zipfile.ZipFile(raft/'models.zip') as z:z.extract('models/raft-things.pth',raft)
    link=RUN/'torch_cache/hub/checkpoints/dino_vitbase16_pretrain.pth'
    link.parent.mkdir(exist_ok=True,parents=True)
    if not link.exists():link.symlink_to(RUN/'cache/dino_model'/link.name)
