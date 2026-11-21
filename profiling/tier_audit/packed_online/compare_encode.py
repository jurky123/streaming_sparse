import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
out=Path(__file__).parent
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
for last in ('intersection_fast','intersection'):
    names=('dense','base',last)
    sources=[np.load('/tmp/compare_short721_'+name+'.npy',mmap_mode='r') for name in names]
    width,height=2304,552
    title=Image.new('RGB',(width,40),(20,23,28))
    pen=ImageDraw.Draw(title)
    for i,name in enumerate(names):
        pen.text((768*i+384,20),name,fill='white',font=font,anchor='mm')
    canvas=np.empty((height,width,3),np.uint8)
    canvas[:40]=np.asarray(title)
    path=out/f'coffee_dense_base_{last}.mp4'
    ff=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','2304x552','-r','16','-i','-','-c:v','libx264','-threads','4','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    for frame in range(80):
        for i,source in enumerate(sources):
            canvas[40:,i*768:(i+1)*768]=source[frame]
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    assert ff.wait()==0
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]))['streams'][0]
    assert (stream['width'],stream['height'],stream['nb_read_frames'],stream['r_frame_rate'])==(width,height,'80','16/1')
    print(path,stream['duration'],flush=True)
