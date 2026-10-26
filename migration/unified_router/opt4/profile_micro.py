"""Locate remaining kernels in the compact route on the actual cached block."""
import argparse
import gzip
import json
from pathlib import Path
import torch
from ..common import atomic_json
from ..learning import ActionHead
from .route import CompactRoute


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args()
    root=Path(a.directory);out=root/'optimization4/micro_profile';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    d=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=d['q'][:,:512];sketch=d['sketch'];head=ActionHead(d['head']).cuda();head.load_state_dict(d['head']);head.eval()
    route=CompactRoute('packed','fp16','tf32x3');step=q.new_tensor(3.,dtype=torch.float32)
    def run():
        with torch.profiler.record_function('OPT::prepare'):state=route.begin(d['prepared'],sketch)
        with torch.profiler.record_function('OPT::score'):s,raw=route.score(q,state)
        with torch.profiler.record_function('OPT::threshold_candidates'):
            m=route.mask(s,d['beta']);r,a=route.candidates(s,m,sketch.origins,d['beta'],128,1,512)
        with torch.profiler.record_function('OPT::statistics_candidates'):st,cv=route.stats(q,sketch,s,m,r,a,raw,state)
        with torch.profiler.record_function('OPT::actions'):return route.actions(q,sketch,s,m,r,a,d['beta'],step,st,cv,head)
    for _ in range(3):run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:run()
    path=out/'trace.json';prof.export_chrome_trace(str(path))
    events=json.loads(path.read_text())['traceEvents'];rows={}
    for e in events:
        if e.get('cat')=='kernel':
            r=rows.setdefault(e['name'],dict(kernel=e['name'],ms=0.,calls=0,registers=e['args'].get('registers per thread'),shared=e['args'].get('shared memory')))
            r['ms']+=e['dur']/1000;r['calls']+=1
    result=sorted(rows.values(),key=lambda r:-r['ms']);atomic_json(out/'kernels.json',result)
    with gzip.open(out/'trace.json.gz','wb')as f:f.write(path.read_bytes())
    path.unlink();print(json.dumps(result[:15],indent=2),flush=True)


if __name__=='__main__':main()
