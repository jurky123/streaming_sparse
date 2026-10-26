"""Paired graph timings against the user-accepted Opt4 FP16 route."""
import argparse,time
from pathlib import Path
from types import SimpleNamespace
import torch
from ..common import atomic_json
from ..learning import ActionHead
from ..opt2.attention import attention
from ..opt4_fused import install
from ..opt4.route import CompactRoute
from .route import FlashRoute
from .weight_route import WeightRoute
from .select import FusedSelectMixin,fused_select


def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    return graph,out


def timing(graph,reps=100):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/reps

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True)
    for k,v in dict(qm=64,gp=4,br=64,splits=4,cs=16,n=512).items():p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--fused-select',action='store_true');p.add_argument('--method',default='recompute');p.add_argument('--candidate',default='mma');p.add_argument('--energy',default='gram');a=p.parse_args();root=Path(a.directory)
    opts={k:getattr(a,k) for k in ['qm','gp','br','splits','cs','energy','candidate']}
    name=('fused_'if a.fused_select else '')+a.method+'_'+'_'.join(f'{k}{v}'for k,v in opts.items())+f'_n{a.n}'
    out=root/'optimization5/probes';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/(name+'.status.json'),dict(status='compiling',options=opts))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;install()
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:a.n];sketch=data['sketch'];prepared=data['prepared'];beta=data['beta']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    router=SimpleNamespace(kind='K32-Set',beta=beta,heads={'24_K32-Set':head})
    step=q.new_tensor(3.,dtype=torch.float32)
    ref=CompactRoute('streamgram','fp16','fp16',tile=64);route_cls=WeightRoute if a.method=='weights' else FlashRoute
    if a.fused_select:route_cls=type('FusedRoute',(FusedSelectMixin,route_cls),{})
    route=route_cls(**opts)
    rs=ref.begin(prepared,sketch);state=route.begin(prepared,sketch)
    reference=ref.run(24,q,sketch,rs,step,router);actual=route.run(24,q,sketch,state,step,router)
    score,raw=route.score(q,state);expected_score,_=ref.score(q,rs)
    assert torch.isfinite(score).all() and actual.any(-1).all()
    mask=route.mask(score,beta);rem,add=route.candidates(score,mask,sketch.origins,beta,128,1,q.shape[1])
    st,cv=route.stats(q,sketch,score,mask,rem,add,raw,state)
    assert torch.isfinite(st[:,:73]).all() and torch.isfinite(cv).all()
    yref=attention(q,data['k'],data['v'],reference);y=attention(q,data['k'],data['v'],actual)
    ydense=torch.nn.functional.scaled_dot_product_attention(q[None],data['k'][None],data['v'][None])[0]
    nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
    rg,_=capture(lambda:ref.run(24,q,sketch,rs,step,router));ng,_=capture(lambda:route.run(24,q,sketch,state,step,router))
    pg,_=capture(lambda:route.begin(prepared,sketch))
    # Raise clocks, then alternate ordering in balanced pairs.
    for _ in range(4):timing(rg,150);timing(ng,150)
    rt=[];nt=[]
    for i in range(6):
        if i%2:nt.append(timing(ng));rt.append(timing(rg))
        else:rt.append(timing(rg));nt.append(timing(ng))
    components={}
    if a.fused_select:
        fg,_=capture(lambda:fused_select(score,sketch.origins,beta));components['fused_select']=timing(fg)
    for label,fn in [('score',lambda:route.score(q,state)),('stats',lambda:route.stats(q,sketch,score,mask,rem,add,raw,state)),('mask',lambda:route.mask(score,beta)),('candidates',lambda:route.candidates(score,mask,sketch.origins,beta,128,1,q.shape[1])),('actions',lambda:route.actions(q,sketch,score,mask,rem,add,beta,step,st,cv,head))]:
        cg,_=capture(fn);components[label]=timing(cg)
    result=dict(options=dict(method=a.method,fused_select=a.fused_select,**opts),name=name,head_queries=q.shape[0]*q.shape[1],
        changed_queries=int((actual!=reference).any(-1).sum()),changed_bits=int((actual!=reference).sum()),
        score_max_abs=float((score-expected_score).abs().max()),score_nmse=nmse(score,expected_score),
        attention_nmse_vs_reference=nmse(y,yref),attention_nmse_vs_dense=nmse(y,ydense),reference_attention_nmse_vs_dense=nmse(yref,ydense),
        keep=float((actual*sketch.costs).sum()/(actual.shape[0]*actual.shape[1]*sketch.costs.sum())),
        reference_keep=float((reference*sketch.costs).sum()/(reference.shape[0]*reference.shape[1]*sketch.costs.sum())),
        reference_route_ms=sum(rt)/len(rt),route_ms=sum(nt)/len(nt),reference_samples=rt,samples=nt,prepare_ms=timing(pg),components_ms=components)
    
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])as prof:
        route.stats(q,sketch,score,mask,rem,add,raw,state)
    result['stats_kernels']=[dict(name=e.key,ms=e.device_time_total/1000)for e in prof.key_averages()if e.device_type==torch.autograd.DeviceType.CUDA]
    atomic_json(out/(name+'.json'),result);print(result,flush=True)
    atomic_json(out/(name+'.status.json'),dict(status='completed'))

if __name__=='__main__':main()
