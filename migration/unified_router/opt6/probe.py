"""Balanced CUDA-graph timing of the head and whole route on real inputs."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from ..common import atomic_json
from ..learning import ActionHead
from ..opt2.attention import attention
from ..opt5.network import OptimizedRoute,OPTIONS
from ..opt5.select import fused_select
from ..opt5.probe import capture,timing
from .route import HeadRoute

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--method',default='packed');p.add_argument('--warps',type=int,default=4);p.add_argument('--pair',default='tf32x3');p.add_argument('--n',type=int,default=512)
    a=p.parse_args();root=Path(a.directory);out=root/'optimization6/probes';out.mkdir(parents=True,exist_ok=True)
    opts=dict(method=a.method,warps=a.warps,pair=a.pair);name=f'{a.method}_w{a.warps}_{a.pair}_n{a.n}'
    atomic_json(out/(name+'.status.json'),dict(status='compiling',options=opts))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:a.n];sketch=data['sketch'];prepared=data['prepared'];beta=data['beta']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    router=SimpleNamespace(kind='K32-Set',beta=beta,heads={'24_K32-Set':head});step=q.new_tensor(3.,dtype=torch.float32)
    ref=OptimizedRoute(**OPTIONS);route=HeadRoute(**opts);state=ref.begin(prepared,sketch)
    score,raw=ref.score(q,state);mask,rem,add=fused_select(score,sketch.origins,beta)
    st,cv=ref.stats(q,sketch,score,mask,rem,add,raw,state)
    def action(r):return r.actions(q,sketch,score,mask,rem,add,beta,step,st,cv,head)
    reference=action(ref);actual=action(route);assert actual.any(-1).all()
    yref=attention(q,data['k'],data['v'],reference);y=attention(q,data['k'],data['v'],actual)
    rg,_=capture(lambda:action(ref));ng,_=capture(lambda:action(route))
    rfull,_=capture(lambda:ref.run(24,q,sketch,state,step,router));nfull,_=capture(lambda:route.run(24,q,sketch,state,step,router))
    for _ in range(4):timing(rfull,150);timing(nfull,150)
    samples=[]
    for i in range(6):
        graphs=[('reference_head',rg),('head',ng),('reference_route',rfull),('route',nfull)]
        if i%2:graphs.reverse()
        samples.append({k:timing(g)for k,g in graphs})
    result=dict(name=name,options=opts,queries=q.shape[0]*q.shape[1],changed_queries=int((actual!=reference).any(-1).sum()),
        attention_delta_nmse=float((y.float()-yref.float()).square().sum()/yref.float().square().sum()),
        timings_ms={k:sum(r[k]for r in samples)/len(samples)for k in samples[0]},samples=samples)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:action(route)
    result['kernels']=[dict(name=e.key,ms=e.device_time_total/1000)for e in prof.key_averages()if e.device_type==torch.autograd.DeviceType.CUDA]
    atomic_json(out/(name+'.json'),result);print(result,flush=True)
    atomic_json(out/(name+'.status.json'),dict(status='completed'))

if __name__=='__main__':main()
