import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import lpips
from local_experiments.high_sparse_experiment import prefix_quality

d=Path(__file__).resolve().parent
metric=lpips.LPIPS(net='alex').eval().requires_grad_(False)
@torch.inference_mode()
def score(a,b):
    result={}
    for n in ([80] if len(a)==80 else [80,480]):
        if np.array_equal(a[:n],b[:n]):
            result[n]=dict(psnr='inf',lpips=0.,ssim=1.)
            continue
        rows=[]
        for i in range(0,n,80):
            def pad(x):
                return [np.concatenate([x[:21],x[i:i+80]])]
            rows.append(prefix_quality(pad(a),pad(b),metric)['per_frame'])
        val={k:np.mean([x[k] for x in rows]).item() for k in rows[0]}
        result[n]=dict(psnr=-10*math.log10(val['mse']),lpips=val['lpips'],ssim=val['ssim'])
    return result
def load(name):return np.load(d/(name+('_5s' if '--short' in sys.argv else '')+'.npy'),mmap_mode='r')
ref=load('dense_before')
output=d/('quality_partial.json' if '--partial' in sys.argv else 'quality_5s.json' if '--short' in sys.argv else 'quality.json')
report={}
for m in ('dense','base','lightforcing','intersection_fast'):
    if '--partial' in sys.argv and m!='dense' and not (d/(m+'_after.mp4')).exists(): continue
    a=ref if m=='dense' else load(m+'_before')
    b=ref if m=='dense' else load(m+'_after')
    report[m]={k:score(x,y) for k,x,y in [('before_vs_dense',a,ref),('after_vs_dense',b,ref),('after_vs_before',b,a)]}
    output.write_text(json.dumps(report,indent=2))
