"""Union of device execution intervals per stream for one warmed nine-chunk trajectory."""
import argparse
import json
from pathlib import Path
import tempfile
import torch
from local_experiments.video_runtime_v1.accepted_tiers import config
from local_experiments.video_runtime_v1.deployment import configure,load_weights,make_runtime
from local_experiments.video_runtime_v1.weights import ROOT
from local_experiments.broad_common import Engine
from local_experiments.rollout_suite import generate_chunk

def union(intervals):
    total=0
    end=0
    for start,stop in sorted(intervals):
        total+=max(0,stop-max(start,end))
        end=max(end,stop)
    return total

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dense',action='store_true',help='Profile the matched native Dense backbone')
    args=ap.parse_args()
    cfg=config('base',local_assets='/tmp/streaming_sparse-0/runtime')
    if args.dense:
        cfg.mode='dense_native'
    configure(cfg)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine()
    engine.wrapper.requires_grad_(False)
    if args.dense:
        from local_experiments.video_runtime_v1.backbone_ops import install
        install(engine,cfg)
        rt=None
    else:
        rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
    from adapter import Prepared
    rt.packed_attention=Prepared()
    rt.set_cuda_fixed15(True)
    rt.packed_attention.enabled=True
    emb=torch.load(ROOT/'artifacts/e24_joint_representatives_v1/embeddings/city_walk.pt',weights_only=True).cuda()
    for c in range(9):generate_chunk(engine,emb,701,c)
    torch.cuda.synchronize()
    (engine if args.dense else rt).reset()
    rows=[]
    with tempfile.TemporaryDirectory(dir='/tmp') as folder:
        file=Path(folder)/'chunk.json'
        for c in range(9):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                generate_chunk(engine,emb,701,c)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(file))
            events=[e for e in json.loads(file.read_text())['traceEvents']
                    if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('dur',0)>0]
            intervals=[(e['ts'],e['ts']+e['dur']) for e in events]
            active=union(intervals)
            span=max(b for a,b in intervals)-min(a for a,b in intervals)
            kernels={}
            for e in events:
                name=e['name']
                kernels[name]=kernels.get(name,0)+e['dur']
            row=dict(chunk=c,span_ms=span/1000,active_ms=active/1000,idle_ms=(span-active)/1000,
                     concurrent_excess_ms=(sum(b-a for a,b in intervals)-active)/1000,
                     events=len(events),streams=len({e['args'].get('stream') for e in events}),
                     by_name_ms={n:us/1000 for n,us in sorted(kernels.items(),key=lambda x:-x[1])})
            print({k:v for k,v in row.items() if k!='by_name_ms'},flush=True)
            rows.append(row)
    if rt is not None:rt.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
