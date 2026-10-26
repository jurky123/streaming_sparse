"""Freeze main-training best/last checkpoints and merge full Val/video quality."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from migration.paired_router import runtime
from streaming_sparse.v08.common import atomic_json,sha,tensor_digest,source_identity
from streaming_sparse.v08.metrics import summarize
from streaming_sparse.v08.replay import ReplayDataset
from migration.frozen_quality.report import paired


def prepare(root,stage):
    experiment,spec,_=runtime.configuration(root,stage);directory=Path(spec['directory']);quality=directory/'quality'
    if(quality/'manifest.json').exists():return
    quality.mkdir(exist_ok=True);ds=ReplayDataset('val',require_complete=True)
    source=dict(mass=directory/'mass.pt'if stage=='dense0'else directory/'runs/seed0/warmstart_K/best.pt',
        Klast=directory/'runs/seed0/F-K/last.pt',Vlast=directory/'runs/seed0/F-V/last.pt',Kbest=directory/'runs/seed0/F-K/best.pt',Vbest=directory/'runs/seed0/F-V/best.pt')
    info={};lora=None
    for name,path in source.items():
        value=torch.load(path,map_location='cpu',weights_only=True);dst=quality/'checkpoints'/f'{name}.pt';dst.parent.mkdir(exist_ok=True)
        shutil.copy2(path,dst);digest=tensor_digest(value['lora'].items())
        if lora is None:lora=digest
        assert digest==lora and all(torch.count_nonzero(v)==0 for n,v in value['lora'].items()if n.endswith('lora_B'))
        info[name]=dict(path=str(dst),sha256=sha(dst),information=value['information'],updates=value['updates'],router_sha256=tensor_digest(value['router'].items()))
    assert info['Klast']['updates']==info['Vlast']['updates']
    atomic_json(quality/'manifest.json',dict(source_sha256=source_identity(),stage=stage,checkpoints=info,val_fingerprint=ds.fingerprint(),
        val_inputs=len(ds),prompt_ids=sorted(ds.registry['splits']['val']),generation_seeds=[0,1],videos=True,
        sparse_layers=spec['layers'],dense_layers=[i for i in range(30)if i not in spec['layers']],keep=.2,backbone_and_lora_frozen=True,
        primary_comparison=['Klast','Vlast'],test_used=False,full_budget_updates=info['Klast']['updates']))


def merge(root,stage):
    experiment,spec,_=runtime.configuration(root,stage);quality=Path(spec['directory'])/'quality';manifest=json.loads((quality/'manifest.json').read_text())
    names=['Dense','mass','Klast','Vlast','Kbest','Vbest'];offline={};videos={}
    for name in names:
        rows=[r for rank in range(8)for r in json.loads((quality/f'rank{rank}/{name}_offline.json').read_text())['rows']]
        assert len(rows)==192;offline[name]=summarize(rows)
    rows=[r for rank in range(8)for r in json.loads((quality/f'rank{rank}/video_rows.json').read_text())];assert len(rows)==16
    for name in names[1:]:
        values=[r['metrics'][name]['quality']['sparse_frames_only']for r in rows];mse=float(np.mean([v['mse_rgb01']for v in values]))
        videos[name]=dict(lpips=float(np.mean([v['lpips']for v in values])),psnr_db=-10*math.log10(mse)if mse else None,
            prompts={pid:float(np.mean([r['metrics'][name]['quality']['sparse_frames_only']['lpips']for r in rows if r['prompt_id']==pid]))for pid in manifest['prompt_ids']},
            chunks=[dict(chunk=i+1,lpips=float(np.mean([r['metrics'][name]['quality']['chunks'][i]['lpips']for r in rows])))for i in range(4)])
    result=dict(passed=True,manifest=manifest,offline=offline,videos=videos,video_cases=rows,
        paired_last_offline=paired(offline['Klast']['prompts'],offline['Vlast']['prompts']),
        paired_last_video_lpips=paired(videos['Klast']['prompts'],videos['Vlast']['prompts']))
    atomic_json(quality/'summary.json',result)
    text=[f'# 冻结骨干配对 Router：{stage}','',f"两组主训练实际更新：{manifest['full_budget_updates']}。稀疏层：{spec['layers']}；其余Dense。8个Val Prompt×2Seed；480p、4chunk。",'',
        '骨干和LoRA全冻结；K/V同预算、同数据顺序。主指标在公共x0 Replay输入上计算；视频PSNR/LPIPS排除首个Dense chunk，与同噪声Dense比较。',
        '', '| 模型 | 离线NMSE↓ | 后36帧PSNR↑ | 后36帧LPIPS↓ |','|---|---:|---:|---:|']
    for name in names[1:]:text.append(f"| {name} | {offline[name]['prompt_equal_error']:.9f} | {videos[name]['psnr_db']:.4f} | {videos[name]['lpips']:.6f} |")
    text+=['',f"末尾权重离线配对：{result['paired_last_offline']}",f"末尾权重视频配对：{result['paired_last_video_lpips']}",'',
        '仅一个Router训练Seed，开发Val结论；达到最大预算不等于已证明收敛。精确停止判据与两组平台情况见各自stop_decision.json。', '', '## 全部视频','']
    for row in sorted(rows,key=lambda r:(r['prompt_id'],r['generation_seed'])):
        pid=f"{row['prompt_id']}.seed{row['generation_seed']}";text.append(f"- {pid}：[Dense / Klast / Vlast](videos/{pid}/paired_Dense_Klast_Vlast.mp4)；[Dense / Kbest / Vbest](videos/{pid}/current_Dense_Kbest_Vbest.mp4)")
    (quality/'REPORT.md').write_text('\n'.join(text)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--stage',required=True);p.add_argument('--merge',action='store_true');a=p.parse_args()
    (merge if a.merge else prepare)(Path(a.directory).resolve(),a.stage)


if __name__=='__main__':main()
