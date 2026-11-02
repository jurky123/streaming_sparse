"""Stop at first actual full-rollout divergence; retain only tiny diagnostic arrays."""
import json,math
from pathlib import Path
import torch
from .broad_common import Engine
from .rollout_suite import generate_chunk
from .route_whole_runtime import WholeRuntime
from .route_graph_runtime import GraphRuntime,RouteGraph
from .route_norm_kernels import logz
from .route_fused_kernels import _score
from .high_sparse_gain import score
from .learned_cache.io import atomic
import triton
ROOT=Path('artifacts/route_kernel_20261031/debug')
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=Engine();engine.shared_route_graph=RouteGraph();rt=WholeRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt');base=rt.route;count=0
    def route(l,q,p):
        nonlocal count
        fast=rt.router;rt.router=engine.shared_route_graph;old=GraphRuntime.route(rt,l,q,p).clone();pk=rt.active_pk.clone();code=rt.active_codes.clone();meta=rt.route_meta
        rt.router=fast;new=base(l,q,p);count+=1
        diff=int((old!=new).any(-1).sum());pd=int((pk!=rt.active_pk).sum());cd=int((code!=rt.active_codes).sum())
        if diff or pd or cd:
            total=p+72;mass,sf,mask,own,add,drop=meta
            raw=engine.shared_route_graph.raw[:12*288*total*4].view(12,288,total*4)
            native_logz=(raw/math.sqrt(128)).reshape(12,72,4,total,4).logsumexp(-1)+math.log(16)
            prob=native_logz.softmax(-1);weighted=mass*mask;den=weighted.sum(-1,keepdim=True).clamp_min(1e-12);selected=weighted@code.float()/den;allvalue=mass@code.float();out=torch.empty_like(mass)
            _score[(864,)](prob,mass,code,fast.vv,fast.kvar,fast.weight,mask,den,selected,allvalue,out,total,triton.next_power_of_2(total),num_warps=4,enable_fp_fusion=False)
            ref=score(mass,sf,rt.sw[l]);tiles=(old!=new).any(-1)[0].nonzero();details=[]
            for h,t in tiles.tolist():
                details.append(dict(h=h,t=t,old=old[0,h,t].tolist(),new=new[0,h,t].tolist(),score=ref[h,t].tolist(),fused_score=out[h,t].tolist(),mask=mask[h,t].tolist(),own=int(own[h,t])))
            h,t=tiles[0].tolist()
            atomic(ROOT/'boundary.pt',dict(prob=prob[h:h+1,t:t+1].contiguous().cpu(),mass=mass[h:h+1,t:t+1].contiguous().cpu(),code=code[h:h+1].cpu(),vv=fast.vv[:12*total].view(12,total)[h:h+1].cpu(),kv=fast.kvar[:12*total].view(12,total)[h:h+1].cpu(),w=rt.sw[l][h:h+1].cpu(),mask=mask[h:h+1,t:t+1].contiguous().cpu(),den=den[h:h+1,t:t+1].contiguous().cpu(),sel=selected[h:h+1,t:t+1].contiguous().cpu(),allvalue=allvalue[h:h+1,t:t+1].contiguous().cpu(),sf=sf[h:h+1,t:t+1].contiguous().cpu(),score=ref[h:h+1,t:t+1].contiguous().cpu()),True)
            data=dict(event=engine.current,layer=l,history=p,calls=count,different_tiles=diff,pk_diff=pd,code_diff=cd,logz_diff=int((native_logz!=logz(raw,total)).sum()),max_score_error=float((ref-out).abs().max()),details=details)
            atomic(ROOT/'first_divergence.json',data);print(json.dumps({k:v for k,v in data.items() if k!='details'}),flush=True)
            raise RuntimeError('First divergence captured')
        return new
    rt.route=route
    try:
        emb=torch.load('artifacts/mass_variance_20261030/embeddings/basketball_court.pt',weights_only=True).cuda()
        for c in range(41):generate_chunk(engine,emb,1001,c);print('chunk',c,'calls',count,flush=True)
    finally:rt.close()
if __name__=='__main__':main()
