import argparse,io,json,random,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
root=Path(__file__).parent

def extract(path,frames):
    expression='+'.join(f'eq(n,{f})' for f in frames)
    w,h=248,166
    cmd=['ffmpeg','-loglevel','error','-threads','2','-i',str(path),'-vf',f"select='{expression}',scale={w}:{h}",'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-threads','2','-']
    buf=subprocess.check_output(cmd);length=w*h*3
    assert len(buf)==len(frames)*length
    return [Image.frombytes('RGB',(w,h),buf[i*length:(i+1)*length]) for i in range(len(frames))]

def sheets(scene,idx,early):
    folder=root/'visual_review';folder.mkdir(exist_ok=True)
    variants=['original','C_only'] if early else ['original','C_only','reader_only','trained']
    random.Random(98173+71*scene+idx).shuffle(variants)
    base=folder/f'{scene}_{idx}_{"partial" if early else "full"}'
    base.with_suffix('.json').write_text(json.dumps(dict(zip('ABCD',variants))))
    sets={'sequence':[0,20,48,96,160,240,336,416,479],'first':[18,19,20,21,22,23,24,27],'roll':[66,67,68,69,70,71,72,75]}
    for name,frames in sets.items():
        sheet=Image.new('RGB',(30+len(frames)*248,22+len(variants)*166),'#202020')
        pen=ImageDraw.Draw(sheet)
        for j,n in enumerate(frames):pen.text((34+j*248,4),f'f{n}   {n/16:.2f}s',fill='white')
        for i,v in enumerate(variants):
            line=extract(root/f'{scene}_{idx}_{v}.mp4',frames)
            pen.text((8,80+i*166),'ABCD'[i],fill='white')
            for j,im in enumerate(line):sheet.paste(im,(30+j*248,22+i*166))
        sheet.save(str(base)+'_'+name+'.jpg',quality=91)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
    for scene in (0,1):
        for idx in (11,21):sheets(scene,idx,a.partial)
