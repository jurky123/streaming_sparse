"""Replay the historical 18-frame LightForcing entry on the current input."""
import json
import os
import sys
import time
from pathlib import Path

import torch

folder = Path(__file__).resolve().parent
repo = folder.parents[1]
sys.path.insert(0, str(repo / 'reproduction/vbench_current'))
from local_experiments.broad_common import noise
from light import build

torch.set_num_threads(2)
with torch.inference_mode():
    pipe = build(41, window=18)
    pipe.text_encoder.embeds = torch.load(repo.parent / '.tools/streaming-sparse-vbench-smoke/embeddings/short_721.pt', weights_only=True).cuda()
    init = torch.cat([noise(701, c, 0) for c in range(41)], 1)
    phases = [noise(701,c,p).flatten(0,1) for c in range(41) for p in (1,2,3)]
    import wan.modules.sparse_attention as sa
    sparse = sa._attention.apply
    hits = []
    def observed(*args):
        hits.append((args[0].shape[1],args[1].shape[1],args[5]))
        return sparse(*args)
    sa._attention.apply = observed
    times = []
    for rep in range(4):
        events = {}
        def timed(fn, name):
            def wrapper(*args, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn(*args, **kwargs)
                end.record()
                events.setdefault(name, []).append((start,end))
                return result
            return wrapper
        if rep == 3:
            for block in pipe.generator.model.blocks:
                att = block.self_attn
                att.causal_rope_fun = timed(att.causal_rope_fun, 'query_and_window_rope')
            cat = torch.cat
            cat_timed = timed(cat, 'history_kv_cat')
            def measured_cat(inputs,*args,**kwargs):
                fn = cat_timed if len(inputs)==2 and inputs[0].ndim==4 and inputs[0].shape[-2:]==(12,128) else cat
                return fn(inputs,*args,**kwargs)
            torch.cat = measured_cat
        calls = iter(phases)
        original = torch.randn_like
        hits.clear()
        torch.randn_like = lambda *a,**kw: next(calls)
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            _, z = pipe.inference(init,['cached'],return_latents=True,profile=False)
        finally:
            torch.randn_like = original
        torch.cuda.synchronize()
        wall = time.monotonic()-start
        assert next(calls,None) is None and len(hits)==6150 and z.isfinite().all()
        if rep < 3:
            times.append(wall)
        stats = {name: [len(pairs),sum(a.elapsed_time(b) for a,b in pairs)/1000] for name,pairs in events.items()}
        data = dict(times=times, calls=len(hits), main_tflop=sum(4*12*q*128*k*64 for q,_,k in hits)/1e12, events_count_seconds=stats)
        print(data,flush=True)
        (folder/'official_sxm.json').write_text(json.dumps(data))
