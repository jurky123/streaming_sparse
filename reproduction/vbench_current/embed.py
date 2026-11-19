from pathlib import Path
import os,torch,json,time
from utils.wan_wrapper import WanTextEncoder
from paths import RUN as r
torch.set_num_threads(4)
with torch.inference_mode():
    model=WanTextEncoder().to(device='cuda',dtype=torch.bfloat16)
    for kind,file in [('long','prompts/MovieGenVideoBench_extended.txt'),('short','prompts/vbench/all_dimension.txt')]:
        prompts=Path(file).read_text().splitlines()
        if kind=='long':prompts=prompts[:128]
        for i in range(0,len(prompts),4):
            batch=prompts[i:i+4]
            z=model(batch)['prompt_embeds'].cpu()
            for j in range(len(batch)):
                path=r/'embeddings'/f'{kind}_{i+j}.pt'
                torch.save(z[j:j+1].clone(),path.with_suffix('.tmp'))
                os.replace(path.with_suffix('.tmp'),path)
            if i%64==0:print(kind,i,tuple(z.shape),flush=True)
