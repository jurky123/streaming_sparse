"""Isolate cached log mass against the deployed compiled expression."""
import argparse,json
import torch
from .weights import ROOT
from .profile_route import capture
from .benchmark import measure
from .tensor_bmm import bf16_inputs_bmm
from . import logmass_cached as lm
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--block',type=int,default=256);ap.add_argument('--warps',type=int,default=4);ap.add_argument('--pages',type=int,default=432);ap.add_argument('--fused',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);lm.BLOCK=a.block;lm.WARPS=a.warps;lm.FUSED=a.fused
    out=ROOT/f'artifacts/video_runtime_v1/logmass_cached_v1/p{a.pages}_b{a.block}_w{a.warps}_f{int(a.fused)}';out.mkdir(parents=True,exist_ok=True)
    e=torch.load(ROOT/'artifacts/video_runtime_v1/fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
    q=e['q'].cuda().index_select(2,e['offsets'].cuda());k=e['summary']['keys'][:,:a.pages].cuda().float();h,t,m,d=q.shape
    z=bf16_inputs_bmm(q.flatten(1,2),k.flatten(1,2).transpose(1,2)).reshape(h,t,m,a.pages,4)/(d**.5)
    def baseline():
        x=z.double().logsumexp(-1);return x-x.logsumexp(-1,keepdim=True)
    funcs={'baseline':baseline,'cached':lambda:lm.normalized(z)};graphs={};values={};rows={}
    for n,f in funcs.items():
        funcs[n]=torch.compile(f,fullgraph=True,options={'triton.cudagraphs':False});graphs[n],values[n]=capture(funcs[n])
    for _ in range(30):
        for g in graphs.values():g.replay()
    for n,g in graphs.items():rows[n]=measure(g.replay,50)
    x,y=values.values();result=dict(rows=rows,bitwise=torch.equal(x,y),max_abs=float((x-y).abs().max()),nmse=float((x-y).square().sum()/x.square().sum()),speedup=rows['baseline']['gpu_ms_median']/rows['cached']['gpu_ms_median'],shape=list(z.shape))
    for n,f in funcs.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:f();torch.cuda.synchronize()
        (out/f'{n}_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=15))
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
