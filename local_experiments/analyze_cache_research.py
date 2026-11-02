"""Static research figures, raw metrics, and synchronized video comparisons."""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import time
import av
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .analyze_official_baselines import video_data,load_dino,features
from .run_kv_history import write_json,sha

METHODS=['dense','spatial','k','kv1','k_tail','kv1_tail']
LABELS=['Dense','Spatial','K-only','Joint-KV','K + correction','KV + correction']
CASES=['sailboat','skateboard','dog']


def contact_sheet(folder,thumbs):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    canvas=Image.new('RGB',(7*192,6*158+26),'white');draw=ImageDraw.Draw(canvas)
    for col,t in enumerate([0,5,10,15,20,25,29.94]):draw.text((col*192+6,3),f'{t:g} s',font=font,fill='black')
    for row,method in enumerate(METHODS):
        y=row*158+26;draw.text((5,y+2),LABELS[row],font=font,fill='black')
        for col,im in enumerate(thumbs[method]):canvas.paste(im,(col*192,y+28))
    canvas.save(folder/'contact_sheet.png')


def comparison(folder):
    path=folder/'comparison.mp4'
    if path.exists():return
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    with ExitStack() as stack:
        readers=[stack.enter_context(av.open(str(folder/f'{m}.mp4'))).decode(video=0) for m in METHODS]
        writer=stack.enter_context(imageio.get_writer(path,fps=16,codec='libx264',pixelformat='yuv420p',
            macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18','-threads','4']))
        for i in range(480):
            canvas=Image.new('RGB',(3*384,2*284),(20,20,20));draw=ImageDraw.Draw(canvas)
            for j,reader in enumerate(readers):
                x,y=(j%3)*384,(j//3)*284
                draw.text((x+5,y+5),LABELS[j],font=font,fill='white')
                im=Image.fromarray(next(reader).to_ndarray(format='rgb24')).resize((384,256))
                canvas.paste(im,(x,y+28))
            writer.append_data(np.asarray(canvas))
        assert all(next(r,None) is None for r in readers)


def plot_screen(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=list(csv.DictReader((root/'screen/metrics.csv').open()))
    def mean_for(key,**filters):
        values=[float(r[key]) for r in frame if r[key] and all(r[k]==str(v) for k,v in filters.items())]
        return float(np.mean(values))
    out=root/'analysis';out.mkdir(exist_ok=True)
    arms=[('spatial','mass'),('k','mass'),('kv1','mass'),('k','tail'),('k','var_tail'),('kv1','var_tail')]
    labels=['Spatial','K','KV','K + mean tail','K + var tail','KV + var tail']
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,key,title in zip(axs,['nmse','output_mean_bias'],['Attention-output NMSE','Mean-output bias / reference energy']):
            for ci,case in enumerate(CASES):
                y=[mean_for(key,case=case,group=g,mode=m) for g,m in arms]
                ax.plot(np.arange(len(arms)),y,marker=['o','s','^'][ci],linestyle=['-','--',':'][ci],
                    color=['#0072B2','#A23B00','#006B56'][ci],label=case)
            ax.set_xticks(range(len(arms)),labels,rotation=25,ha='right');ax.set_title(title);ax.set_ylim(bottom=0)
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Real QKV: 3 prompts, 1 seed each; matched historical budget (including correction)')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'mechanism.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for mode,color,marker in [('mass','#0072B2','o'),('var_tail','#A23B00','s')]:
            for ax,axis in zip(axs,['layer','step']):
                values=sorted(set(int(r[axis]) for r in frame))
                means=[mean_for('nmse',group='kv1',mode=mode,**{axis:x}) for x in values]
                ax.plot(values,means,marker=marker,color=color,label=mode)
                ax.set_xlabel(axis);ax.set_ylabel('Mean output NMSE');ax.set_ylim(bottom=0);ax.legend();ax.grid(alpha=.2)
        fig.suptitle('Joint-KV error by layer / denoising step; descriptive repeated observations')
        fig.savefig(out/'layer_step.png',dpi=180,facecolor='white');plt.close(fig)
    keys=['nmse','output_mean_bias','history_mass_coverage','mask_churn']
    with (out/'mechanism_by_case.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['case','group','mode']+keys);writer.writeheader()
        for case,group,mode in sorted(set((r['case'],r['group'],r['mode']) for r in frame)):
            writer.writerow(dict(case=case,group=group,mode=mode,
                **{k:mean_for(k,case=case,group=group,mode=mode) for k in keys}))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/cache_research_20261029')
    p.add_argument('--wait',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
    root=Path(a.root);out=root/'analysis';out.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    plot_screen(root)
    model,weights=load_dino(a.device);rows=[];video_hashes={}
    for case in CASES:
        folder=root/'video'/case;thumbs={}
        for method in METHODS:
            done=folder/f'{method}.result.json'
            if a.wait and method!='dense':
                print(json.dumps(dict(phase='waiting',case=case,method=method)),flush=True)
                while not done.exists():time.sleep(5)
            if method=='dense':done=Path('artifacts/ar_official_baselines_20261029')/case/'dense.result.json'
            result=json.loads(done.read_text());assert result['completed']
            data,samples,thumbs[method]=video_data(folder/f'{method}.mp4')
            fpath=out/f'{case}.{method}.dino.npy'
            if fpath.exists():f=np.load(fpath)
            else:f=features(model,samples,a.device);np.save(fpath,f)
            np.savez_compressed(out/f'{case}.{method}.frames.npz',**data)
            video_hashes[f'{case}/{method}']=sha(folder/f'{method}.mp4')
            y,c=data['brightness'][69:],data['chroma'][69:]
            motion=data['flow'][data['flow'][:,0]>=4.3125,1]
            research=result.get('research',{})
            row=dict(case=case,method=method,
                brightness_step_mean_pp=float(np.abs(np.diff(y)).mean()*100),
                dark_dip_max_pp=float(data['dark_dip'][69:].max()*100),
                dark_dip_frames_over_2pp=int((data['dark_dip'][69:]>.02).sum()),
                chroma_drift=float(np.linalg.norm(c-data['chroma'][:9].mean(0),axis=1).mean()),
                flow_mean=float(motion.mean()),dino_adjacent=float((f[6:]*f[5:-1]).sum(-1).mean()),
                dino_to_first=float((f[5:]*f[0]).sum(-1).mean()),
                latent_nmse=research.get('latent_nmse',0.),
                main_kv_gib=result['max_self_kv_storage_bytes']/2**30,
                aux_permutation_mib=research.get('persistent_auxiliary_permutation_peak_bytes',0)/2**20,
                generation_seconds=result['generation_seconds'],
                peak_allocated_gib=result['peak_allocated_bytes']/2**30,
                preprocess_gpu_seconds=research.get('preprocess_cuda_ms',0)/1000)
            rows.append(row);print(json.dumps(row),flush=True)
            with (out/'video_metrics.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        contact_sheet(folder,thumbs);comparison(folder)
    write_json(out/'video_summary.json',dict(rows=rows,methods=METHODS,cases=CASES,
        definitions={'interval':'RGB frames 69..479, same as previous official baseline report.',
        'brightness':'Mean RGB weighted .2126/.7152/.0722; adjacent change in pp.',
        'dark_dip':'Positive difference below centered 9-frame median; threshold 2 pp.',
        'chroma':'Spatial mean RGB / channel sum, mean L2 distance from own first 9 frames; content confounds.',
        'flow':'Farneback at 192x128 and 4fps, px/s; motion and camera effects, not quality.',
        'dino':'ViT-B/16 full-image cosine at 1fps; not official VBench; still images can score high.',
        'timing':'Shared GPU; dense reuses earlier run; unsuitable for end-to-end speedup claims.',
        'statistics':'3 prompts x 1 seed; no independence assumption across heads/steps, no significance claims.'}))
    write_json(out/'provenance.json',dict(script_sha256=sha(__file__),dino_sha256=sha(weights),video_sha256=video_hashes,
        figures='PNG/PDF/SVG static; no HTML; fixed-size downsampling only, no color adjustments.',
        sample_unit='Prompt; all three prompts shown individually, no confidence intervals.'))
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':9,'pdf.fonttype':42}):
        fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        panels=[('brightness_step_mean_pp','Brightness change (pp)'),('dark_dip_max_pp','Maximum short dark dip (pp)'),
          ('chroma_drift','Global chroma drift'),('flow_mean','Optical flow (px/s)'),('dino_adjacent','DINO adjacent cosine'),('latent_nmse','Latent NMSE vs dense')]
        for ax,(key,title) in zip(axs.flat,panels):
            for ci,case in enumerate(CASES):
                ys=[next(r[key] for r in rows if r['case']==case and r['method']==m) for m in METHODS]
                ax.plot(np.arange(6),ys,marker=['o','s','^'][ci],linestyle=['-','--',':'][ci],
                    color=['#0072B2','#A23B00','#006B56'][ci],label=case)
            ax.set_xticks(range(6),LABELS,rotation=28,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
            if key!='dino_adjacent':ax.set_ylim(bottom=0)
        axs[0,0].legend(fontsize=8);fig.suptitle('30-second closed-loop videos; 3 prompts x 1 seed; no overall quality ranking')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'video_metrics.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
    print(json.dumps(dict(completed=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
