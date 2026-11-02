"""Independent early-video descriptors and viewable comparison videos."""
import json,os
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import torch
import av,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .analyze_official_baselines import load_dino,features
from .stream_eval_common import ROOT
from .learned_cache.io import atomic,digest,disk_guard
from .gap_peer_watch import PeerWatch
@torch.no_grad()
def main():
    manifest=json.loads((ROOT/'manifest.json').read_text());assert json.loads((ROOT/'status.json').read_text())['status']=='complete'
    torch.set_num_threads(4);model,weights=load_dino('cuda');rows=[]
    with PeerWatch() as watch:
        for case in manifest['cases']:
            for method in manifest['methods']:
                folder=ROOT/case/method;dest=folder/'early_descriptors.json'
                if dest.exists():rows.append(json.loads(dest.read_text()));continue
                images=[];rgb=[];motion=[];prev=None
                with av.open(str(folder/'video.mp4')) as r:
                    for i,f in enumerate(r.decode(video=0)):
                        if i>=101:break
                        a=f.to_ndarray(format='rgb24');small=a[::4,::4].astype(np.float32)/255
                        if i>=21:
                            rgb.append(small.mean((0,1)));motion.append(float(np.abs(small-prev).mean()))
                            if (i-21)%8==0:images.append(Image.fromarray(a))
                        prev=small
                feat=features(model,images,'cuda');adj=(feat[1:]*feat[:-1]).sum(-1);first=(feat[1:]*feat[0]).sum(-1)
                brightness=np.array(rgb)@np.array([.2126,.7152,.0722]);padded=np.pad(brightness,4,mode='edge');med=np.array([np.median(padded[i:i+9]) for i in range(len(brightness))])
                row=dict(case=case,method=method,window=[21,101],sample_step=8,dino_adjacent_cosine=float(adj.mean()),dino_first_cosine=float(first.mean()),brightness_step_pp=float(np.abs(np.diff(brightness)).mean()*100),dark_dip_max_pp=float(np.maximum(med-brightness,0).max()*100),motion_l1=float(np.mean(motion)),weights_sha256=digest(weights),scope='DINO image-feature consistency and temporal proxies on encoded early video; not official VBench scores, no text-alignment claim, frozen video can score highly.')
                atomic(dest,row);rows.append(row)
    atomic(ROOT/'descriptor_peers.json',watch.summary());watch.validate();del model;torch.cuda.empty_cache()
    atomic(ROOT/'early_descriptors.json',rows)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for case in manifest['cases']:
        dest=ROOT/case/'comparison.mp4'
        if dest.exists():continue
        disk_guard(ROOT,50,1);temp=dest.with_suffix('.tmp.mp4')
        with ExitStack() as stack:
            readers=[stack.enter_context(av.open(str(ROOT/case/m/'video.mp4'))).decode(video=0) for m in manifest['methods']]
            writer=stack.enter_context(imageio.get_writer(temp,fps=16,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_log_level='error',output_params=['-crf','18','-threads','4']))
            for i in range(480):
                canvas=Image.new('RGB',(1152,568),(20,20,20));draw=ImageDraw.Draw(canvas)
                for j,(m,reader) in enumerate(zip(manifest['methods'],readers)):
                    x,y=(j%3)*384,(j//3)*284;draw.text((x+6,y+3),m,font=font,fill='white');canvas.paste(next(reader).to_image().resize((384,256)),(x,y+28))
                draw.text((774,320),f'{case}  {i/16:.2f}s',font=font,fill='white');writer.append_data(np.array(canvas))
        temp.replace(dest)
    from .stream_eval_report import main as report
    report()
    path=ROOT/'report.zh.md'
    with path.open('a') as f:
        f.write('\n## 前段独立视频描述指标\n\n这些辅助指标不能取代视频判断；DINO一致性高也可能来自运动停滞。\n\n|场景|方法|DINO相邻余弦↑|DINO首帧余弦↑|运动L1|最大暗闪/百分点↓|\n|---|---|---:|---:|---:|---:|\n')
        for r in rows:f.write(f"|{r['case']}|{r['method']}|{r['dino_adjacent_cosine']:.4f}|{r['dino_first_cosine']:.4f}|{r['motion_l1']:.4f}|{r['dark_dip_max_pp']:.3f}|\n")
    atomic(ROOT/'quality_status.json',dict(status='complete',units=len(rows)))
if __name__=='__main__':main()
