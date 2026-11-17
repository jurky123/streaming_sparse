"""Single captured city input: exact+tail reference versus streamed mixture."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .weights import ROOT
from .attention import Workspace,exact,compensate_tensor,merge
from .tensor_bmm import tensor_bmm
from .stream_attention import full_values,mixed_attention
from .benchmark import measure

@torch.no_grad()
def run(padded=False,tail_mode=1,warps=4):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rt,e,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt');route=rt.compute(e,True);s=e['s'];q=e['q'];h,t,n,d=q.shape;p=s.mean.shape[1]
    path=next((ROOT/'artifacts/full_coverage_v1/data/city_walk_s701').glob('c8_l15_s0_h0_t*.pt'));ev=torch.load(path,weights_only=True);kv=torch.load(path.parent/ev['kv_file'],weights_only=True)
    rawk=kv['k'].cuda()[None].expand(h,-1,-1,-1).contiguous();rawv=kv['v'].cuda()[None].expand_as(rawk).contiguous()
    nq=q.flatten(1,2).bfloat16().transpose(0,1).contiguous()[None];nk=rawk.flatten(1,2).transpose(0,1).contiguous()[None];nv=rawv.flatten(1,2).transpose(0,1).contiguous()[None]
    ids,mask=route[0],route[2]
    if padded:
        from ..review_v1.ops import Summary
        parts=[]
        for name in ('keys','mean','detail','stats'):
            oldpart=getattr(s,name);newpart=torch.empty((h,432,*oldpart.shape[2:]),device='cuda',dtype=oldpart.dtype)
            newpart[:,:p].copy_(oldpart);parts.append(newpart[:,:p])
        s=Summary(*parts)
    work=Workspace();tail=torch.compile(compensate_tensor,fullgraph=True,options={'triton.cudagraphs':False})
    def old():
        z=tensor_bmm(q.flatten(1,2),s.keys.float().flatten(1,2).transpose(1,2)).reshape(h,t,n,p,4)/d**.5
        oe,le=exact(nq,nk,nv,ids,work);oa,la=tail(z,s,e['basis'],mask,e['haar_float']);return merge(oe,le,oa,la)
    values=full_values(s,e['basis'])
    if padded:
        storage=torch.empty(h,432,4,128,device='cuda',dtype=values.dtype);storage[:,:p].copy_(values);values=storage[:,:p]
    go,x=capture(old);before=measure(go.replay,25);rows=[]
    for splits in (1,2,4,8):
        gn,y=capture(lambda:mixed_attention(nq,nk,nv,s,values,ids,mask,work,splits,tail_mode,warps))
        delta=(x.float()-y.float());rows.append(dict(splits=splits,nmse=float(delta.square().sum()/x.float().square().sum()),max_abs=float(delta.abs().max()),finite=bool(torch.isfinite(y).all()),after=measure(gn.replay,25)))
        print(splits,rows[-1]['nmse'],rows[-1]['after']['gpu_ms_median'],flush=True)
    path=Path('artifacts/video_runtime_v1/ieee_compensation');path.mkdir(parents=True,exist_ok=True);(path/(f'stream_attention_padded_mode{tail_mode}_w{warps}.json' if padded else f'stream_attention_mode{tail_mode}_w{warps}.json')).write_text(json.dumps(dict(padded=padded,tail_mode=tail_mode,warps=warps,kernel_stats={str(k):v for k,v in __import__("local_experiments.video_runtime_v1.stream_attention",fromlist=["KERNEL_STATS"]).KERNEL_STATS.items()},before=before,rows=rows,identity=identity,scope='Resident page Values; includes shared Q64 QK in old, excludes one-time page reconstruction in new'),indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--padded',action='store_true');p.add_argument('--warps',type=int,choices=[4,8],default=4);p.add_argument('--tail-mode',type=int,choices=[0,1,3,4],default=1);a=p.parse_args();run(a.padded,a.tail_mode,a.warps)
