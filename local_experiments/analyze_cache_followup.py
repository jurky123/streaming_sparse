"""Analyze selective correction and preserve the all-layer failure diagnosis."""
import csv
import argparse
import json
from pathlib import Path
import time
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .analyze_official_baselines import video_data,load_dino,features
from .run_kv_history import write_json


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plots-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);root=Path('artifacts/cache_research_20261029');out=root/'analysis'
    model=None if args.plots_only else load_dino('cuda')[0];rows=[]
    for case in ([] if args.plots_only else ['sailboat','skateboard','dog']):
        folder=root/'followup'/case
        while not (folder/'k_guard.result.json').exists():time.sleep(5)
        result=json.loads((folder/'k_guard.result.json').read_text())
        data,samples,thumbs=video_data(folder/'k_guard.mp4')
        feat=features(model,samples,'cuda');np.save(out/f'{case}.k_guard.dino.npy',feat)
        np.savez_compressed(out/f'{case}.k_guard.frames.npz',**data)
        y,c=data['brightness'][69:],data['chroma'][69:];motion=data['flow'][data['flow'][:,0]>=4.3125,1]
        research=result['research']
        row=dict(case=case,method='k_guard',brightness_step_mean_pp=float(np.abs(np.diff(y)).mean()*100),
            dark_dip_max_pp=float(data['dark_dip'][69:].max()*100),dark_dip_frames_over_2pp=int((data['dark_dip'][69:]>.02).sum()),
            chroma_drift=float(np.linalg.norm(c-data['chroma'][:9].mean(0),axis=1).mean()),flow_mean=float(motion.mean()),
            dino_adjacent=float((feat[6:]*feat[5:-1]).sum(-1).mean()),dino_to_first=float((feat[5:]*feat[0]).sum(-1).mean()),
            latent_nmse=research['latent_nmse'],main_kv_gib=result['max_self_kv_storage_bytes']/2**30,
            aux_permutation_mib=research['persistent_auxiliary_permutation_peak_bytes']/2**20,
            generation_seconds=result['generation_seconds'],peak_allocated_gib=result['peak_allocated_bytes']/2**30,
            preprocess_gpu_seconds=research['preprocess_cuda_ms']/1000)
        rows.append(row);print(json.dumps(row),flush=True)
        with (out/'followup_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        # Wait for the original K-only arm before making a matched contact sheet.
        while not (root/'video'/case/'k.result.json').exists():time.sleep(5)
        canvas=Image.new('RGB',(7*192,3*158+26),'white');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
        for col,t in enumerate([0,5,10,15,20,25,29.94]):draw.text((col*192+6,3),f'{t:g} s',font=font,fill='black')
        for ri,(method,title) in enumerate([('dense','Dense'),('k','K-only'),('k_guard','K + correction at layer 8')]):
            path=(folder if method=='k_guard' else root/'video'/case)/f'{method}.mp4'
            draw.text((4,ri*158+29),title,font=font,fill='black')
            with av.open(str(path)) as reader:
                col=0
                for i,frame in enumerate(reader.decode(video=0)):
                    if i in [0,80,160,240,320,400,479]:
                        canvas.paste(Image.fromarray(frame.to_ndarray(format='rgb24')).resize((192,128)),(col*192,ri*158+54));col+=1
        canvas.save(folder/'contact_sheet.png')
    if rows:write_json(out/'followup_summary.json',dict(rows=rows,calibration_case='sailboat',held_out_for_layer_selection=['skateboard','dog']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((root/'diagnostic/layer_selection.json').read_text())['summary']
    with plt.rc_context({'font.size':10,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
        for ci,group in enumerate(['k','kv1']):
            s=[x for x in summary if x['group']==group];x=[r['layer'] for r in s]
            for key,label,color,marker in [('mass_nmse','Sparse mass','#0072B2','o'),('tail_nmse','Variance + tail','#A23B00','s')]:
                axes[0,ci].plot(x,[r[key] for r in s],label=label,color=color,marker=marker,markersize=3)
            axes[0,ci].set(yscale='log',title=group+' / all 12 heads',ylabel='Mean output NMSE (log scale)');axes[0,ci].legend()
            axes[1,ci].plot(x,[r['tail_mass_mae_p95'] for r in s],color='#0072B2',marker='o',markersize=3,label='p95 over head / step / chunk')
            axes[1,ci].axhline(.05,color='#A23B00',linestyle='--',label='predeclared 0.05 threshold')
            axes[1,ci].set(xlabel='Layer',ylabel='Tail probability error',ylim=(0,1));axes[1,ci].legend(fontsize=8)
        fig.suptitle('Failure diagnosis on sailboat: all layers and heads; no independent-sample significance claim')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'layer_diagnosis.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
    print(json.dumps(dict(completed=True)),flush=True)


if __name__=='__main__':main()
