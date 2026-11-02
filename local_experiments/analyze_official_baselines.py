"""Budget, timing, video stability, motion and DINO descriptors; not VBench scores."""
import argparse
from contextlib import ExitStack
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types
import time

import av
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter
import torch
import torch.nn.functional as F

METHODS = ["dense", "lightforcing", "dasa", "forcingkv"]
from .run_kv_history import write_json, sha

LABELS = ['Dense', 'Light Forcing', 'DASA*', 'Forcing-KV']
SHORT = ['Dense sliding window', 'Light Forcing / same weights', 'DASA / full inference', 'Forcing-KV / native budget']


def video_data(path):
    rgb, samples, gray, flow = [], [], None, []
    thumbs = []
    with av.open(str(path)) as reader:
        stream = reader.streams.video[0]
        assert float(stream.average_rate) == 16
        for index, frame in enumerate(reader.decode(video=0)):
            pixels = frame.to_ndarray(format='rgb24')
            rgb.append(pixels.mean((0,1))/255.)
            if index % 16 == 0:
                samples.append(Image.fromarray(pixels).copy())
            if index in [0, 80, 160, 240, 320, 400, 479]:
                thumbs.append(Image.fromarray(pixels).resize((192,128)))
            if index % 4 == 0:
                now = cv2.cvtColor(cv2.resize(pixels, (192,128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
                if gray is not None:
                    field = cv2.calcOpticalFlowFarneback(gray, now, None, .5, 3, 15, 3, 5, 1.2, 0)
                    flow.append((index/16, float(np.linalg.norm(field, axis=2).mean()) * 4))
                gray = now
    assert len(rgb) == 480
    rgb = np.array(rgb)
    brightness = rgb @ np.array([.2126,.7152,.0722])
    chroma = rgb / np.maximum(rgb.sum(-1,keepdims=True), 1e-12)
    dips = np.maximum(0., median_filter(brightness,size=9,mode='nearest')-brightness)
    return dict(rgb=rgb, brightness=brightness, chroma=chroma, dark_dip=dips, flow=np.array(flow)), samples, thumbs


def load_dino(device):
    directory = Path('/data/chenjiayu/zhikun/VBench/models/facebookresearch_dino_main')
    # DINO's source imports `utils`; inject only its required initialization
    # helper and immediately restore the generation project's namespace.
    old = sys.modules.get('utils')
    shim = types.ModuleType('utils')
    shim.trunc_normal_ = torch.nn.init.trunc_normal_
    sys.modules['utils'] = shim
    try:
        spec = importlib.util.spec_from_file_location('evaluation_dino_vit', directory/'vision_transformer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old is None:
            sys.modules.pop('utils', None)
        else:
            sys.modules['utils'] = old
    model = module.vit_base(patch_size=16, num_classes=0)
    weights = Path('/data/chenjiayu/zhikun/VBench/models/checkpoints/dino_vitbase16_pretrain.pth')
    state = torch.load(weights,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False), weights


@torch.no_grad()
def features(model, samples, device):
    from torchvision import transforms as T
    transform = T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    values = []
    for start in range(0,len(samples),8):
        batch = torch.stack([transform(im) for im in samples[start:start+8]]).to(device)
        values.append(F.normalize(model(batch).float(),dim=-1).cpu().numpy())
    return np.concatenate(values)


def montage(folder, thumbnails):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    canvas = Image.new('RGB',(7*192,4*158+26),'white')
    draw = ImageDraw.Draw(canvas)
    for i,t in enumerate([0,5,10,15,20,25,29.94]):
        draw.text((i*192+6,3),f'{t:g} s',font=font,fill='black')
    for row, method in enumerate(METHODS):
        y = row*158+26
        draw.text((5,y+3),SHORT[row],font=font,fill='black')
        for col,im in enumerate(thumbnails[method]):canvas.paste(im,(col*192,y+28))
    canvas.save(folder/'contact_sheet.png')


def comparison(folder):
    path = folder/'comparison.mp4'
    if path.exists():return
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    with ExitStack() as stack:
        readers = [stack.enter_context(av.open(str(folder/f'{m}.mp4'))).decode(video=0) for m in METHODS]
        writer = stack.enter_context(imageio.get_writer(path,fps=16,codec='libx264',pixelformat='yuv420p',
            macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18','-threads','4']))
        for i in range(480):
            canvas = Image.new('RGB',(768,568),(20,20,20));draw=ImageDraw.Draw(canvas)
            for j,reader in enumerate(readers):
                x,y=(j%2)*384,(j//2)*284
                draw.text((x+5,y+5),SHORT[j],font=font,fill='white')
                frame=next(reader).to_ndarray(format='rgb24')
                canvas.paste(Image.fromarray(frame).resize((384,256)),(x,y+28))
            writer.append_data(np.asarray(canvas))
        assert all(next(r,None) is None for r in readers)


def trace_metrics(folder, method):
    trace=json.loads((folder/f'{method}.attention_trace.json').read_text())
    denoise=[r for r in trace if r['timestep']>0]
    pairs=0
    for r in denoise:
        k=r['k_tokens']
        if r['sparse'] and r['k_tokens']!=r['q_tokens']:
            k=min(k//64,int((1-r['requested_sparsity'])*(k//64)))*64
        pairs+=r['q_tokens']*k*r['heads']
    chunks=[]
    for line in (folder/f'{method}.log').read_text().splitlines():
        if line.startswith('{"phase": "chunk_complete"'):
            chunks.append(json.loads(line))
    assert len(chunks)==41
    return dict(attention_pairs_vs_full_cap=pairs/(41*4*30*12*4608*27648),
                steady_chunk_seconds_median=float(np.median([r['seconds'] for r in chunks if r['chunk']>=7])),
                observed_attention_calls=len(trace),
                sparse_attention_calls=sum(r['sparse'] for r in trace),
                denoise_reduced_query_calls=sum(r['q_tokens']<4608 for r in denoise))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment');p.add_argument('--cases',nargs='+',default=['sailboat','skateboard','dog'])
    p.add_argument('--device',default='cpu');p.add_argument('--wait',action='store_true',help='Wait for each scene to finish while the generation suite runs');a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    root=Path(a.experiment);out=root/'analysis';out.mkdir(exist_ok=True)
    model,weights=load_dino(a.device);rows=[]
    manifest=json.loads((root/'manifest.json').read_text());cap=manifest['total_self_kv_storage_cap_bytes']
    for case in a.cases:
        folder=root/case;all_data,thumbs={},{};reference_latent=None
        if a.wait:
            print(json.dumps({'phase':'waiting_for_scene','case':case}),flush=True)
            while not all((folder/f'{m}.result.json').exists() for m in METHODS):time.sleep(10)
        for method in METHODS:
            result=json.loads((folder/f'{method}.result.json').read_text())
            assert result['completed'] and result['chunks']==41
            assert result['checkpoint_sha256']==manifest['checkpoints']
            assert result['shared_inputs_sha256']==sha(folder/'shared_inputs.pt')
            cache=json.loads((folder/f'{method}.cache_trace.json').read_text())
            assert all(r['self_kv_storage_bytes']<=cap for r in cache)
            latent=torch.load(folder/f'{method}.latents.pt',map_location='cpu',weights_only=True)
            assert torch.isfinite(latent).all()
            if reference_latent is None:reference_latent=latent[:,:3].float().clone()
            prefix_difference=float((latent[:,:3].float()-reference_latent).abs().max())
            del latent
            data,samples,thumbs[method]=video_data(folder/f'{method}.mp4')
            dino_path=out/f'{case}.{method}.dino.npy'
            if dino_path.exists():f=np.load(dino_path)
            else:f=features(model,samples,a.device);np.save(dino_path,f)
            np.savez_compressed(out/f'{case}.{method}.frames.npz',**data);all_data[method]=data
            y,c=data['brightness'][69:],data['chroma'][69:]
            motion=data['flow'][data['flow'][:,0]>=4.3125,1]
            row=dict(case=case,method=method,
                brightness_step_mean_pp=float(np.abs(np.diff(y)).mean()*100),
                brightness_step_p95_pp=float(np.quantile(np.abs(np.diff(y)),.95)*100),
                dark_dip_max_pp=float(data['dark_dip'][69:].max()*100),
                dark_dip_frames_over_2pp=int((data['dark_dip'][69:]>.02).sum()),
                chroma_step_mean=float(np.linalg.norm(np.diff(c,axis=0),axis=1).mean()),
                flow_mean_pixels_per_second=float(motion.mean()),
                flow_p95_pixels_per_second=float(np.quantile(motion,.95)),
                dino_adjacent_1s_cosine=float((f[6:]*f[5:-1]).sum(-1).mean()),
                dino_to_first_frame_cosine=float((f[5:]*f[0]).sum(-1).mean()),
                max_kv_gib=result['max_self_kv_storage_bytes']/2**30,
                auxiliary_mib=result['max_auxiliary_bytes']/2**20,
                peak_allocated_gib=result['peak_allocated_bytes']/2**30,
                generation_seconds=result['generation_seconds'],
                initial_chunk_latent_max_abs_difference_from_dense=prefix_difference,
                **trace_metrics(folder,method))
            rows.append(row);print(json.dumps(row),flush=True)
        ref=all_data['dense']['chroma'][:9].mean(0)
        for row in rows:
            if row['case']==case:
                row['chroma_drift_from_dense_opening_mean']=float(np.linalg.norm(all_data[row['method']]['chroma'][69:]-ref,axis=1).mean())
                own=all_data[row['method']]['chroma'][:9].mean(0)
                row['chroma_drift_from_own_opening_mean']=float(np.linalg.norm(all_data[row['method']]['chroma'][69:]-own,axis=1).mean())
        montage(folder,thumbs);comparison(folder)
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    definitions=dict(
        frame_interval='RGB frames 69..479 (4.3125..29.9375 seconds), identical for all methods. Official arithmetic/position policies can cause initial chunk differences, which are measured rather than assumed zero.',
        brightness='Spatial mean nonlinear RGB code values, weighted .2126/.7152/.0722; absolute adjacent-frame change in percentage points.',
        dark_dip='Positive difference below centered 9-frame median brightness, nearest padding. Fixed >2pp threshold.',
        chroma='Spatial mean RGB divided by channel sum. Drift from own opening and from dense opening (frames 0..8) both reported. Content changes also affect it.',
        motion='Farneback optical flow on 192x128 grayscale at 4fps; mean magnitude times 4, pixels/sec in resized image. Includes camera motion and flicker; not VBench dynamic_degree.',
        dino='DINO ViT-B/16 global image features at 1fps, ImageNet normalization, short-side 256 resize / 224 center crop. Not object-tracked or a VBench score; static outputs can score high.',
        timing='Generation excludes checkpoint loading and decoding, includes first-shape compilation. Steady statistic is median of chunks 7..41. Shared GPU; no isolated speedup claims.',
        kv='Unique physical tensor storage for all persistent self-KV across layers, including current-chunk buffers, DASA clean KV and Forcing-KV warmup copies. Excludes temporary operator tensors and text cross-KV; peak allocated memory and auxiliary state reported separately.',
        attention='Logical denoising token-pair counts from operand shapes; Light Forcing uses official 64-token top-k block formula. Denominator: 41 chunks * 4 denoising steps * 30 layers * 12 heads * 4608 queries * 27648 keys. Includes warmup underfill. Excludes routing, clean-cache updates, FFN and cross-attention; not a speedup.',
        replication='3 prompts, 1 seed per prompt. Plotted points are individual scenes, not independent timing repeats. No confidence intervals or aggregate quality ranking.',
        limitations='Brightness, chroma, DINO and flow are descriptive proxies affected by composition and motion. Inspect videos together. Same weights, original complete inference pipelines, documented compatibility changes; no additional rebase experiment.')
    write_json(out/'summary.json',dict(rows=rows,definitions=definitions))
    write_json(out/'provenance.json',dict(script_sha256=sha(__file__),dino_weights=str(weights),
        dino_sha256=sha(weights),method_order=METHODS,cases=a.cases,
        video_sha256={f'{case}/{method}':sha(root/case/f'{method}.mp4') for case in a.cases for method in METHODS}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('max_kv_gib','Persistent self-KV incl. workspace','GiB'),
            ('steady_chunk_seconds_median','Observed time (shared GPU)','Seconds / chunk'),
            ('brightness_step_mean_pp','Global brightness changes','Mean absolute change (pp)'),
            ('dark_dip_max_pp','Largest short dark dip','Brightness difference (pp)'),
            ('flow_mean_pixels_per_second','Motion proxy (not a quality ranking)','Flow magnitude (pixels/s)'),
            ('dino_adjacent_1s_cosine','Adjacent-frame visual similarity','DINO cosine (1 s interval)')]
    with plt.rc_context({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        for ax,(key,title,ylabel) in zip(axes.flat,panels):
            for i,case in enumerate(a.cases):
                data=[next(r[key] for r in rows if r['case']==case and r['method']==m) for m in METHODS]
                ax.plot(np.arange(4)+(i-1)*.13,data,['o','s','^'][i],color=['#0072B2','#D55E00','#009E73'][i],label=case)
            ax.set(title=title,ylabel=ylabel,xticks=np.arange(4),xticklabels=LABELS)
            ax.tick_params(axis='x',labelrotation=20);ax.set_ylim(bottom=0)
            if key=='dino_adjacent_1s_cosine':ax.set_ylim(min(.5,min(r[key] for r in rows)-.05),1.01)
            ax.grid(axis='y',alpha=.2)
        axes[0,0].axhline(cap/2**30,color='#555555',linestyle='--',linewidth=1,label='Common storage cap')
        axes[0,0].set_ylim(top=cap/2**30*1.15);axes[0,0].legend(fontsize=8)
        fig.suptitle('Same Self-Forcing weights and inputs; 30 s; 3 prompts × 1 seed\nDASA*: bounded-cache repair; Forcing-KV: smaller native cache; shared-GPU timings',fontsize=12)
        fig.savefig(out/'comparison.png',dpi=180,facecolor='white');fig.savefig(out/'comparison.svg',facecolor='white');plt.close(fig)
        fig,axes=plt.subplots(len(a.cases),2,figsize=(13,3*len(a.cases)),layout='constrained',squeeze=False)
        for ci,case in enumerate(a.cases):
            for mi,m in enumerate(METHODS):
                d=np.load(out/f'{case}.{m}.frames.npz');style=['-','--','-.',':'][mi]
                axes[ci,0].plot(np.arange(480)/16,d['brightness']*100,style,label=LABELS[mi],linewidth=1)
                axes[ci,1].plot(d['flow'][:,0],d['flow'][:,1],style,label=LABELS[mi],linewidth=1)
            axes[ci,0].set(title=case+' / global brightness',ylabel='RGB code-value brightness (%)',xlabel='Seconds')
            axes[ci,1].set(title=case+' / motion proxy',ylabel='Flow magnitude (pixels/s)',xlabel='Seconds')
            for ax in axes[ci]:ax.set_ylim(bottom=0);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.savefig(out/'time_series.png',dpi=180,facecolor='white');plt.close(fig)
    (out/'description.txt').write_text('Six panels show physical self-KV storage, observed shared-GPU time, brightness volatility, dark dips, motion and DINO similarity. Each method has three scene observations. Methods are the original complete inference pipelines; hardware/backend/weight differences from paper settings are documented. No confidence intervals. Read summary.json for definitions.\n')


if __name__=='__main__':main()
