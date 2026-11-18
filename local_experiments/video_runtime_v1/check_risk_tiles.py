"""Time page tilings of the deployed inlined variance and probability computation."""
import argparse
import json
import statistics
import torch
import triton
from .risk_online import kernel, matched_normalizer
from .profile_route import capture
from .weights import ROOT,load
from .tensor_bmm import scaled_route_qk


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cuda-pages',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    root=ROOT/'artifacts/video_runtime_v1'
    model,*_=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt')
    data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
    offset=torch.linspace(0,63,16,device='cuda').round().long()
    q=data['q'].cuda()[0].transpose(0,1).reshape(12,72,64,128).index_select(2,offset)
    u=q.float().square().reshape(12,72,16,8,16).sum(-1)/128
    qn=q.float().norm(dim=-1)/(128**.5)
    rows=[]
    for p in (216,288,360,432):
        st=data['summary']['stats'][:,:p].cuda().contiguous()
        k=data['summary']['keys'][:,:p].cuda().contiguous()
        z=scaled_route_qk(q.flatten(1,2),k.flatten(1,2).transpose(1,2)).reshape(12,72,16,p,4)
        norm=matched_normalizer(z)
        bp=st[:,:,1,11].float()
        c=(bp+bp.amax(-1,keepdim=True)).contiguous()
        variants=[]
        configs=((16,8),(0,0)) if args.cuda_pages else ((16,8),(32,4),(32,8),(64,4),(64,8),(128,8))
        for tile,warps in configs:
            out=torch.empty(12,72,p,device=q.device)
            def call():
                if not tile:
                    from .risk_pages import run
                    return run(u,qn,z,norm,st,model[15].theta,c,out)
                return kernel[(12*72,triton.cdiv(p,tile))](u,qn,z,norm,st,model[15].theta,c,out,
                     72,16,p,tile,False,True,True,False,True,num_warps=warps,enable_fp_fusion=False)
            compiled=call()
            graph,_=capture(call)
            variants.append((graph,out,dict(tile=tile,warps=warps,registers=compiled.n_regs if compiled else None,
                                           spills=compiled.n_spills if compiled else None,shared=compiled.metadata.shared if compiled else None)))
        samples=[[] for _ in variants]
        for rep in range(32):
            for j in range(len(variants)):
                i=(rep+j)%len(variants)
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(15):variants[i][0].replay()
                b.record();b.synchronize()
                if rep>=2:samples[i].append(a.elapsed_time(b)/15)
        for (_,out,info),times in zip(variants,samples):
            info.update(pages=p,ms=statistics.median(times),equal=torch.equal(out,variants[0][1]),
                        max_abs=float((out-variants[0][1]).abs().max()))
            print(info,flush=True)
            rows.append(dict(**info,samples_ms=times))
    folder=root/'risk_tiles_v1'
    folder.mkdir(exist_ok=True)
    (folder/('cuda_pages.json' if args.cuda_pages else 'module.json')).write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
