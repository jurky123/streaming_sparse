"""Budget, timing, video stability, motion and DINO descriptors; not VBench scores."""
import argparse
from contextlib import ExitStack
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

import av
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter
import torch
import torch.nn.functional as F

from .baseline_methods import METHODS
from .run_kv_history import write_json, sha

LABELS = ['FIFO', 'Anchor', 'Rebase', 'Light Forcing\ntransfer', 'DASA-KV\ncapped', 'Forcing-KV\nnative budget']
SHORT = ['FIFO: latest 5 chunks', 'Anchor: first + latest 4', 'Rebase: first + latest 4',
         'Light Forcing: same-weight transfer', 'DASA-KV: capped memory module', 'Forcing-KV: smaller native budget']


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
    canvas = Image.new('RGB',(7*192,6*158+26),(255,255,255))
    draw = ImageDraw.Draw(canvas)
    for i,t in enumerate([0,5,10,15,20,25,29.94]):
        draw.text((i*192+6,3),f'{t:g} s',font=font,fill='black')
    for row, method in enumerate(METHODS):
        y = row*158+26
        draw.text((5,y+3),SHORT[row],font=font,fill='black')
        for col,im in enumerate(thumbnails[method]):
            canvas.paste(im,(col*192,y+28))
    canvas.save(folder/'contact_sheet.png')


def comparison(folder):
    path = folder/'comparison.mp4'
    if path.exists():
        return
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    with ExitStack() as stack:
        readers = [stack.enter_context(av.open(str(folder/f'{m}.mp4'))).decode(video=0) for m in METHODS]
        writer = stack.enter_context(imageio.get_writer(path,fps=16,codec='libx264',pixelformat='yuv420p',
            macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18','-threads','4']))
        for i in range(480):
            canvas = Image.new('RGB',(1152,568),(20,20,20))
            draw = ImageDraw.Draw(canvas)
            for j,reader in enumerate(readers):
                x,y = (j%3)*384,(j//3)*284
                draw.text((x+5,y+5),SHORT[j],font=font,fill='white')
                frame = next(reader).to_ndarray(format='rgb24')
                canvas.paste(Image.fromarray(frame).resize((384,256)),(x,y+28))
            writer.append_data(np.asarray(canvas))
        assert all(next(r,None) is None for r in readers)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment')
    p.add_argument('--cases',nargs='+',default=['sailboat','skateboard','dog'])
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    root=Path(a.experiment)
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    model,weights=load_dino(a.device)
    rows=[]
    for case in a.cases:
        folder=root/case
        all_data,thumbs={},{ }
        reference_latent=None
        for method in METHODS:
            result=json.loads((folder/f'{method}.result.json').read_text())
            assert result['completed'] and result['generated_chunks']==41
            latent=torch.load(folder/f'{method}.latents.pt',map_location='cpu',weights_only=True)
            if reference_latent is None:
                reference_latent=latent[:,:3].clone()
            else:
                torch.testing.assert_close(latent[:,:3],reference_latent,atol=0,rtol=0)
            data,samples,thumbs[method]=video_data(folder/f'{method}.mp4')
            dino_path=out/f'{case}.{method}.dino.npy'
            if dino_path.exists():
                f=np.load(dino_path)
            else:
                f=features(model,samples,a.device)
                np.save(dino_path,f)
            np.savez_compressed(out/f'{case}.{method}.frames.npz',**data)
            all_data[method]=data
            y,c=data['brightness'][69:],data['chroma'][69:]
            motion=data['flow'][data['flow'][:,0]>=4.3125,1]
            # Adjacent 1-second feature pairs; both endpoints after warmup.
            feature_steps=(f[6:]*f[5:-1]).sum(-1)
            row=dict(case=case,method=method,
                brightness_step_mean_pp=float(np.abs(np.diff(y)).mean()*100),
                brightness_step_p95_pp=float(np.quantile(np.abs(np.diff(y)),.95)*100),
                dark_dip_max_pp=float(data['dark_dip'][69:].max()*100),
                dark_dip_frames_over_2pp=int((data['dark_dip'][69:]>.02).sum()),
                chroma_step_mean=float(np.linalg.norm(np.diff(c,axis=0),axis=1).mean()),
                flow_mean_pixels_per_second=float(motion.mean()),
                flow_p95_pixels_per_second=float(np.quantile(motion,.95)),
                dino_adjacent_1s_cosine=float(feature_steps.mean()),
                dino_to_first_frame_cosine=float((f[5:]*f[0]).sum(-1).mean()),
                max_kv_gib=result['max_persistent_kv_bytes']/2**30,
                auxiliary_mib=result['max_auxiliary_state_bytes']/2**20,
                peak_allocated_gib=result['peak_allocated_bytes']/2**30,
                generation_seconds=result['generation_seconds_including_first_shape_compilation'],
                steady_chunk_seconds_median=float(np.median(result['steady_chunk_seconds'])),
                attention_pairs_vs_full_cap=result['denoise_attention_pairs']/result['denoise_full_cap_pairs'])
            rows.append(row)
            print(json.dumps(row),flush=True)
        ref=all_data['anchor']['chroma'][:9].mean(0)
        for row in rows:
            if row['case']==case:
                row['chroma_drift_from_opening_mean']=float(np.linalg.norm(all_data[row['method']]['chroma'][69:]-ref,axis=1).mean())
        montage(folder,thumbs)
        comparison(folder)
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    definitions=dict(
        frame_interval='RGB frames 69..479 (4.3125..29.9375 seconds), identical for all methods. First 3 latent frames verified bitwise identical across all methods.',
        brightness='Spatial mean of nonlinear RGB code values, weighted .2126/.7152/.0722; absolute adjacent-frame change in percentage points.',
        dark_dip='Positive difference below centered 9-frame median brightness, nearest padding. Fixed >2pp threshold.',
        chroma='Spatial mean RGB divided by channel sum. Opening drift uses anchor frames 0..8, same generated initial chunk.',
        motion='Farneback optical flow on 192x128 grayscale at 4fps; mean magnitude times 4, pixels/sec in resized image. Includes camera motion and flicker; not VBench dynamic_degree.',
        dino='DINO ViT-B/16 global image features at 1fps, ImageNet normalization, short-side 256 resize / 224 center crop; adjacent and first-frame cosine. Not object-tracked, not a VBench score; static outputs can score high.',
        timing='Generation includes first-shape compilation. Steady statistic is median of chunks 7..41. Shared GPU, no isolated speedup claims.',
        attention='Exact token-pair counts in attention operands/selected blocks divided by a fully occupied 18-frame extent. Excludes routing, FFN and cross-attention; not wall-time speedup.',
        replication='3 prompts, 1 seed per prompt; plotted points are scenes, not independent timing repeats. No confidence intervals.',
        limitations='Brightness, color, DINO and flow are descriptive proxies affected by motion, composition and lighting; view videos together, do not infer overall quality from one scalar.')
    write_json(out/'summary.json',dict(rows=rows,definitions=definitions))
    write_json(out/'provenance.json',dict(script_sha256=sha(__file__),dino_weights=str(weights),
            dino_sha256=sha(weights),method_order=list(METHODS),cases=a.cases,
            video_sha256={f'{case}/{method}':sha(root/case/f'{method}.mp4') for case in a.cases for method in METHODS}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('max_kv_gib','Persistent historical KV','GiB'),
            ('steady_chunk_seconds_median','Observed generation time (shared GPU)','Seconds / chunk'),
            ('brightness_step_mean_pp','Global brightness changes','Mean absolute change (pp)'),
            ('dark_dip_max_pp','Largest short dark dip','Brightness difference (pp)'),
            ('flow_mean_pixels_per_second','Motion proxy (not a quality ranking)','Flow magnitude (pixels/s)'),
            ('dino_adjacent_1s_cosine','Adjacent-frame visual similarity','DINO cosine (1 s interval)')]
    with plt.rc_context({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ax,(key,title,ylabel) in zip(axes.flat,panels):
            for i,case in enumerate(a.cases):
                data=[next(r[key] for r in rows if r['case']==case and r['method']==m) for m in METHODS]
                ax.plot(np.arange(6)+(i-1)*.13,data,['o','s','^'][i],color=['#0072B2','#D55E00','#009E73'][i],label=case)
            ax.set(title=title,ylabel=ylabel,xticks=np.arange(6),xticklabels=LABELS)
            ax.tick_params(axis='x',labelrotation=25)
            ax.set_ylim(bottom=0 if key!='dino_adjacent_1s_cosine' else .5)
            if key=='dino_adjacent_1s_cosine':
                ax.set_ylim(min(.5, min(r[key] for r in rows)-.05),1.01)
            ax.grid(axis='y',alpha=.2)
        axes[0,0].axhline(4246732800/2**30,color='#555555',linestyle='--',linewidth=1,label='Common KV cap')
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Same Self-Forcing weights and inputs; 30 s; 3 prompts × 1 seed\nMethod adaptations and actual KV use are documented; descriptive screening results',fontsize=12)
        fig.savefig(out/'comparison.png',dpi=180,facecolor='white')
        fig.savefig(out/'comparison.svg',facecolor='white')
        plt.close(fig)
    (out/'description.txt').write_text('Six panels show actual KV storage, observed time, brightness volatility, dark dips, motion and DINO similarity. Each method has three individually plotted scene observations; no confidence interval is estimated. Read summary.json for definitions and limitations.\n')


if __name__=='__main__':
    main()
