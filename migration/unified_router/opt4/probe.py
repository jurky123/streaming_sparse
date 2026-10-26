"""Paired CUDA-graph microbenchmarks and actual attention-error measurements."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from ..common import atomic_json
from ..fast_route import FastRoute
from ..opt2_guarded import guarded_finish
from ..opt2_precision import fma_score
from ..learning import ActionHead
from ..opt2.attention import attention
from .route import CompactRoute


def timed_graph(fn, repeats=100):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=fn()
    for _ in range(5):graph.replay()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)/repeats


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True)
    p.add_argument('--method',choices=['tiled','gram','packed','streamgram'],default='tiled')
    p.add_argument('--precision',default='tf32x3');p.add_argument('--score-precision',default='tf32x3')
    p.add_argument('--gp',type=int,default=8);p.add_argument('--tile',type=int,default=32)
    a=p.parse_args();root=Path(a.directory);out=root/'optimization4/probes';out.mkdir(parents=True,exist_ok=True)
    name=f'{a.method}_{a.precision}_{a.score_precision}_p{a.gp}_q{a.tile}'
    atomic_json(out/(name+'.status.json'),dict(status='loading',options=vars(a)))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:512];sketch=data['sketch'];prepared=data['prepared'];beta=data['beta']
    head=ActionHead(data['head']).cuda();head.load_state_dict(data['head']);head.eval()
    router=SimpleNamespace(kind='K32-Set',beta=beta,heads={'24_K32-Set':head})
    step=q.new_tensor(3.,dtype=torch.float32)
    ref=FastRoute(True,True);ref.score=fma_score;ref.finish=guarded_finish
    route=CompactRoute(a.method,a.precision,a.score_precision,a.gp,a.tile)
    ref.candidates=route.candidates
    state=route.begin(prepared,sketch)
    reference=ref(24,q,sketch,prepared,step,router)
    actual=route.run(24,q,sketch,state,step,router)
    score,raw=route.score(q,state);expected_score,_=fma_score(q,*prepared)
    assert torch.isfinite(score).all() and actual.any(-1).all()
    yref=attention(q,data['k'],data['v'],reference)
    y=attention(q,data['k'],data['v'],actual)
    ydense=torch.nn.functional.scaled_dot_product_attention(q[None],data['k'][None],data['v'][None])[0]
    nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
    result=dict(options=vars(a),name=name,head_queries=q.shape[0]*q.shape[1],
        changed_queries=int((actual!=reference).any(-1).sum()),changed_bits=int((actual!=reference).sum()),
        score_max_abs=float((score-expected_score).abs().max()),score_nmse=nmse(score,expected_score),
        attention_nmse_vs_reference=nmse(y,yref),attention_nmse_vs_dense=nmse(y,ydense),
        reference_attention_nmse_vs_dense=nmse(yref,ydense),
        keep=float((actual*sketch.costs).sum()/(actual.shape[0]*actual.shape[1]*sketch.costs.sum())),
        reference_keep=float((reference*sketch.costs).sum()/(reference.shape[0]*reference.shape[1]*sketch.costs.sum())),
        reference_route_ms=timed_graph(lambda:ref(24,q,sketch,prepared,step,router)),
        route_ms=timed_graph(lambda:route.run(24,q,sketch,state,step,router)),
        prepare_ms=timed_graph(lambda:route.begin(prepared,sketch)) if a.method!='tiled'else 0.)
    atomic_json(out/(name+'.json'),result);print(result,flush=True)
    atomic_json(out/(name+'.status.json'),dict(status='completed'))


if __name__=='__main__':main()
