from pathlib import Path
import json,numpy as np,torch
from local_experiments.stream_eval_common import native_vae,decode,save_video
out=Path(__file__).parent
vae=native_vae()
frames=[np.concatenate(decode(vae,torch.load('/tmp/variable_latent_'+str(i)+'.pt',weights_only=True).cuda(),80)) for i in range(2)]
d=frames[1].astype(np.float32)-frames[0]
print('mean abs',np.abs(d).mean(),'per 16 frames',[round(float(np.abs(c).mean()),3) for c in np.split(d,5)],'psnr',10*np.log10(255**2/(d*d).mean()),flush=True)
save_video(out/'intersection_side_by_side.mp4',[np.concatenate(frames,axis=2)])
