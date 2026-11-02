"""Run official baselines sequentially, then decode using the identical native VAE."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .run_kv_history import ROOT,sha,write_json,status,decode_videos,inspect_video

METHODS=['dense','lightforcing','dasa','forcingkv']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='artifacts/ar_official_baselines_20261029')
    p.add_argument('--cases',nargs='+',default=['sailboat','skateboard','dog'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    a=p.parse_args();os.chdir(ROOT)
    folder=Path(a.output).resolve();folder.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    old=ROOT/'artifacts/ar_sparse_baselines_20261029'
    for case in a.cases:
        target=folder/case;target.mkdir(exist_ok=True)
        for name in ['shared_inputs.pt','inputs.json','prompt.txt']:
            if not (target/name).exists():shutil.copyfile(old/case/name,target/name)
        assert sha(target/'shared_inputs.pt')==json.loads((target/'inputs.json').read_text())['sha256']
    manifest=dict(methods=a.methods,cases=a.cases,date=time.strftime('%Y-%m-%d'),
        height=512,width=768,fps=16,seconds=30,chunks=41,seed=0,
        protocol='Original complete upstream pipelines, shared Self-Forcing EMA weights, shared saved initial/transition noise, shared text embeddings and VAE. No custom rebase arm.',
        historical_kv_budget_frames=15,current_chunk_frames=3,
        total_self_kv_storage_cap_bytes=5096079360,
        budget_explanation='18 frame-equivalents total self-KV storage, accounting for historical cache and current workspace. DASA reserves 15 main-cache + 3 extra clean-KV frames. Dense/LF use 18 main-cache frames. Forcing-KV native smaller caches are reported, not padded.',
        checkpoints=sha(ROOT/'checkpoints/self_forcing_dmd.pt'),
        source_commits={repo:subprocess.check_output(['git','-C',str(ROOT/'external_baselines'/repo),'rev-parse','HEAD'],text=True).strip()
                        for repo in ['LightForcing','DASA','Forcing-KV','DeepForcing']},
        dense_backend='Common PyTorch SDPA',lightforcing_backend='Official sparse Triton kernel, FP32 sum compatibility patch',
        timing_limitation='Shared H200 GPU; descriptive wall times, not isolated speedup measurements.',
        original_method_limitations=['Light Forcing paper includes fine-tuning; controlled same-weight evaluation uses original Self-Forcing weights.',
          'Forcing-KV dynamic temporal mode restricts temporal_context_length to 1..3 frames; original 1/1/1 context configuration uses less than the common cap.',
          'DASA bounded-cache rollover fix: reserve a full chunk, update compressed local pointer, then execute original pruning/reuse branches. Unmodified capped upstream fails at chunk 11; patch and failed log archived.',
          'Runtime paths, frame-token sizing, cache limits, dense backend and saved-input injection are configured. Native algorithmic RoPE is preserved.'])
    write_json(folder/'manifest.json',manifest)
    for repo in ['LightForcing','DASA','Forcing-KV']:
        patch=subprocess.check_output(['git','-C',str(ROOT/'external_baselines'/repo),'diff'],text=True)
        (folder/f'{repo}.compatibility.patch').write_text(patch)
    subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],
                   stdout=(folder/'gpu_at_start.csv').open('w'),check=True)
    for ci,case in enumerate(a.cases):
        target=folder/case
        # Rotate execution order across scenes, keeping one generation job/GPU.
        order=a.methods[ci%len(a.methods):]+a.methods[:ci%len(a.methods)]
        for method in order:
            done=target/f'{method}.result.json'
            if done.exists() and json.loads(done.read_text()).get('completed'):continue
            gen=target/f'{method}.generation.json'
            if not gen.exists():
                status(folder,'generating',case=case,method=method)
                cmd=[sys.executable,str(ROOT/'local_experiments/run_official_baseline.py'),
                     '--method',method,'--inputs',str(target/'shared_inputs.pt'),'--output',str(target)]
                with (target/f'{method}.log').open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            result=json.loads(gen.read_text());assert result['completed'] and result['chunks']==41
            latent=torch.load(target/f'{method}.latents.pt',map_location='cpu',weights_only=True)
            tick=time.perf_counter();decode_videos({method:latent},target,480,16)
            result['decode_seconds']=time.perf_counter()-tick
            result['video']=inspect_video(target/f'{method}.mp4')
            assert result['video']['frames']==480 and result['video']['duration']==30
            write_json(done,result)
            del latent;gc.collect();torch.cuda.empty_cache()
    status(folder,'completed',cases=a.cases,methods=a.methods)


if __name__=='__main__':main()
