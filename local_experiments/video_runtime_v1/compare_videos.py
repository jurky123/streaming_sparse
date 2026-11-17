"""Stream existing encoded videos into a Dense | C+Risk | Full_Pset comparison."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from ..stream_eval_common import save_video
from ..learned_cache.io import atomic

def run(root):
    root=Path(root);outputs=[]
    for folder in sorted((root/'videos').glob('*/*')):
        paths=[folder/method/'video.mp4' for method in ('dense','c_risk','c_pset')]
        if not all(p.exists() for p in paths):continue
        target=folder/'comparison.mp4'
        if target.exists():outputs.append(str(target));continue
        readers=[imageio.get_reader(p,input_params=['-threads','1']) for p in paths];count=[0]
        def batches():
            buf=[]
            for frames in zip(*readers):
                buf.append(np.concatenate(frames,axis=1));count[0]+=1
                if len(buf)==8:yield np.stack(buf);buf=[]
            if buf:yield np.stack(buf)
        try:save_video(target,batches())
        finally:
            for r in readers:r.close()
        expected=105 if folder.name=='chunks9' else 480;assert count[0]==expected
        atomic(folder/'comparison.json',dict(frames=expected,left_to_right=['Dense','C+Risk','C+Risk+Full_Pset'],VAE_decodes=0,source='existing encoded videos; visual comparison only'))
        outputs.append(str(target))
    print(json.dumps(outputs))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
