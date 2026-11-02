"""Generate and decode all declared research arms using shared inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .run_kv_history import ROOT,decode_videos,inspect_video,write_json,sha

ARMS={'spatial':(None,False),'k':(0.,False),'kv1':(1.,False),'k_tail':(0.,True),'kv1_tail':(1.,True)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/cache_research_20261029')
    p.add_argument('--cases',nargs='+',default=['sailboat','skateboard','dog'])
    p.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    a=p.parse_args();root=Path(a.root).resolve();torch.set_num_threads(4)
    assert (root/'prototype_checks.json').exists()
    write_json(root/'video_manifest.json',dict(cases=a.cases,arms=a.arms,
        checkpoint_sha256=sha(ROOT/'checkpoints/self_forcing_dmd.pt'),
        self_kv_cap=5096079360,protocol='video_protocol.zh.md',created=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        code_sha256={n:sha(ROOT/'local_experiments'/n) for n in ['cache_grouping.py','cache_sparse_prototype.py','run_official_baseline.py']}))
    for ci,case in enumerate(a.cases):
        folder=root/'video'/case;folder.mkdir(parents=True,exist_ok=True)
        inputs=ROOT/'artifacts/ar_official_baselines_20261029'/case/'shared_inputs.pt'
        for ext in ['mp4','latents.pt']:
            link=folder/f'dense.{ext}'
            if not link.exists():link.symlink_to(inputs.parent/f'dense.{ext}')
        for name in a.arms[ci%len(a.arms):]+a.arms[:ci%len(a.arms)]:
            target=folder/name;target.mkdir(exist_ok=True)
            weight,tail=ARMS[name]
            write_json(target/'config.json',dict(mode='prototype',name=name,weight=weight,tail=tail,ratio=.2))
            if not (target/'dense.generation.json').exists():
                print(json.dumps(dict(phase='generating',case=case,arm=name)),flush=True)
                cmd=[sys.executable,'-m','local_experiments.run_official_baseline','--method','dense',
                    '--inputs',str(inputs),'--output',str(target),'--research-config',str(target/'config.json')]
                with (target/'run.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not (folder/f'{name}.result.json').exists():
                lat=torch.load(target/'dense.latents.pt',map_location='cpu',weights_only=True)
                decode_videos({name:lat},folder,480,16)
                result=json.loads((target/'dense.generation.json').read_text())
                result['research']=json.loads((target/'research.json').read_text())
                result['video']=inspect_video(folder/f'{name}.mp4');assert result['video']['frames']==480
                write_json(folder/f'{name}.result.json',result)
                del lat
        print(json.dumps(dict(phase='scene_complete',case=case)),flush=True)


if __name__=='__main__':main()
