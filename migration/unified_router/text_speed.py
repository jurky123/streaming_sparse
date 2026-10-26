"""Supplementary native T5 timing, separate from cached-condition generation."""
import argparse
import json
from pathlib import Path
import time
import torch
from .common import ROOT,atomic_json,paired_seed,update


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    folder=out/'text_encoder';folder.mkdir(exist_ok=True)
    try:
        update(folder/'status.json',status='loading')
        paired_seed(0);torch.cuda.set_device(0)
        from utils.wan_wrapper import WanTextEncoder
        registry=json.loads((ROOT/'streaming_sparse/v08/configs/prompt_registry.json').read_text())
        manifest=json.loads((out/'manifest.json').read_text());prompt=registry['texts'][manifest['prompt_id']]
        started=time.perf_counter();encoder=WanTextEncoder().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.synchronize();load_seconds=time.perf_counter()-started
        rows=[]
        for i in range(4):
            update(folder/'status.json',status='warmup'if i==0 else 'benchmark',iteration=i)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            value=encoder([prompt])['prompt_embeds']
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            rows.append(dict(iteration=i,seconds=seconds,shape=list(value.shape),dtype=str(value.dtype),
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        # CPU tokenizer time is diagnostic, not additive to total above.
        token=[]
        for _ in range(10):
            start=time.perf_counter();encoder.tokenizer([prompt],return_mask=True,add_special_tokens=True)
            token.append(time.perf_counter()-start)
        atomic_json(folder/'summary.json',dict(passed=True,load_seconds=load_seconds,warmup=rows[0],samples=rows[1:],
            mean_seconds=sum(x['seconds']for x in rows[1:])/3,tokenizer_mean_seconds=sum(token)/len(token),
            gpu=torch.cuda.get_device_name(0),includes='Native tokenizer, input H2D, T5 and padded prompt embedding; model loading excluded',
            separate_stage=True,not_in_cached_condition_generation_table=True))
        update(folder/'status.json',status='completed')
    except BaseException as error:
        atomic_json(folder/'failed.json',dict(error=repr(error)));raise


if __name__=='__main__':main()
