import json,sys,os,importlib,subprocess
from pathlib import Path
from paths import RUN as r, REPO
import torch
torch.set_num_threads(3)
from vbench.utils import init_submodules
b=Path(sys.argv[1]);payload=json.loads((b/'jobs.json').read_text());kind,d=payload['kind'],payload['dimension']
info=json.loads((r/'info.json').read_text())
rows=[]
for v in payload['videos']:
    path=Path(v['path'])
    if kind=='short':
        row=dict(info[v['i']])
        if d=='human_action':
            # The official evaluator extracts the Kinetics label from the filename.
            alias=b/(v['prompt']+'-'+v['method']+'-'+str(v['seed'])+'.mp4')
            if not alias.exists():alias.symlink_to(path)
            path=alias
        row['video_list']=[str(path)]
    else:
        name=f"{v['method']}-{v['i']}-{v['seed']}"
        link=b/(name+'.mp4')
        if not link.exists():link.symlink_to(path)
        row=dict(prompt_en=v['prompt'],dimension=[d])
        if d!='chunk_discontinuity':
            from segment_cache import ensure_segments
            cache=ensure_segments(link,name)
            split=b/'split_clip';split.mkdir(exist_ok=True)
            dest=split/name
            if not dest.exists():dest.symlink_to(cache,target_is_directory=True)
            row['video_list']=[str(x) for x in sorted(dest.glob('*.mp4'))]
    rows.append(row)
if d=='chunk_discontinuity':
    raft=REPO/'external_baselines/Forcing-KV/evaluation/Raft'
    subs=init_submodules(['dynamic_degree'],local=True)
    subprocess.run([sys.executable,raft/'optical_difference.py','--model',subs['dynamic_degree']['model'],'--path',b,'--device','cuda:0','--k','39','--output_path',b/'raft'],cwd=raft,check=True)
    result=json.loads((b/'raft'/'result_mean.jsonl').read_text().strip())
else:
    j=b/'input.json';j.write_text(json.dumps(rows))
    pkg='vbench' if kind=='short' else 'vbench2_beta_long'
    fn=getattr(importlib.import_module(pkg+'.'+d),('compute_' if kind=='short' else 'compute_long_')+d)
    mods=init_submodules([d],local=True)[d]
    cfg=r/'VBench-master/vbench2_beta_long/configs'
    result=fn(str(j),torch.device('cuda'),mods,imaging_quality_preprocessing_mode='longer',read_frame=False,
        sb_clip2clip_feat_extractor='dino',bg_clip2clip_feat_extractor='clip',w_inclip=1.,w_clip2clip=0.,dev_flag=False,
        slow_fast_eval_config=str(cfg/'slow_fast_params.yaml'),sb_mapping_file_path=str(cfg/'subject_mapping_table.yaml'),
        bg_mapping_file_path=str(cfg/'background_mapping_table.yaml'))
    import numpy as np
    assert np.isfinite(result[0]) and len(result[1])>0, 'empty or nonfinite evaluation'
(b/'result.tmp').write_text(json.dumps(result,default=lambda x:x.item()))
(b/'result.tmp').replace(b/'result.json')
