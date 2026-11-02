"""Fixed, calibration-only layer selection and three selective-correction videos."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .run_kv_history import ROOT,write_json,decode_videos,inspect_video


def select(root):
    rows=list(csv.DictReader((root/'diagnostic/metrics.csv').open()));summary=[];selected={}
    for group in ['k','kv1']:
        selected[group]=[]
        for layer in range(1,30):
            a=[float(r['nmse']) for r in rows if r['group']==group and r['mode']=='mass' and int(r['layer'])==layer]
            rr=[r for r in rows if r['group']==group and r['mode']=='var_tail' and int(r['layer'])==layer]
            b=[float(r['nmse']) for r in rr];e=[float(r['tail_mass_mae']) for r in rr]
            ok=np.mean(b)<=.8*np.mean(a) and np.quantile(e,.95)<=.05 and np.quantile(b,.95)<=np.quantile(a,.95)
            summary.append(dict(group=group,layer=layer,mass_nmse=float(np.mean(a)),tail_nmse=float(np.mean(b)),
                tail_mass_mae_p95=float(np.quantile(e,.95)),tail_nmse_p95=float(np.quantile(b,.95)),mass_nmse_p95=float(np.quantile(a,.95)),
                tail_mass_estimated_mean=float(np.mean([float(r['tail_mass_estimated']) for r in rr])),
                tail_mass_actual_mean=float(np.mean([float(r['tail_mass_actual']) for r in rr])),selected=bool(ok)))
            if ok:selected[group].append(layer)
    result=dict(calibration='sailboat only',criteria='mean tail NMSE <= 0.8 mean mass; tail mass MAE p95 <= .05; tail NMSE p95 <= mass NMSE p95',layers=selected,summary=summary)
    write_json(root/'diagnostic/layer_selection.json',result)
    return selected


def main():
    torch.set_num_threads(4)
    root=ROOT/'artifacts/cache_research_20261029';selected=select(root)
    print(json.dumps(dict(selected_layers=selected)),flush=True)
    for group in ['k','kv1']:
        if not selected[group]:
            print(json.dumps(dict(group=group,skipped='No layer passed the predeclared criterion; identical to mass baseline.')),flush=True)
            continue
        name=group+'_guard'
        for case in ['sailboat','skateboard','dog']:
            folder=root/'followup'/case;folder.mkdir(parents=True,exist_ok=True)
            target=folder/name;target.mkdir(exist_ok=True)
            write_json(target/'config.json',dict(mode='prototype',name=name,weight=0. if group=='k' else 1.,
                tail=True,tail_layers=selected[group],ratio=.2,calibration_case='sailboat',selection='diagnostic_protocol.zh.md'))
            if not (target/'dense.generation.json').exists():
                cmd=[sys.executable,'-m','local_experiments.run_official_baseline','--method','dense',
                    '--inputs',str(ROOT/'artifacts/ar_official_baselines_20261029'/case/'shared_inputs.pt'),
                    '--output',str(target),'--research-config',str(target/'config.json')]
                with (target/'run.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not (folder/f'{name}.result.json').exists():
                latent=torch.load(target/'dense.latents.pt',map_location='cpu',weights_only=True)
                decode_videos({name:latent},folder,480,16)
                result=json.loads((target/'dense.generation.json').read_text());result['research']=json.loads((target/'research.json').read_text())
                result['video']=inspect_video(folder/f'{name}.mp4');assert result['video']['frames']==480
                write_json(folder/f'{name}.result.json',result)
            print(json.dumps(dict(case=case,arm=name,completed=True)),flush=True)


if __name__=='__main__':main()
