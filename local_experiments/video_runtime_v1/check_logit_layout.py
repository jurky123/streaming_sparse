"""Measure route QK, row normalization and Risk on a common tiled logits buffer."""
import argparse
import json
import statistics
import torch
import triton
from .profile_route import capture
from .risk_online import kernel as risk, _normalize
from .tensor_bmm import kernel as qk
from .weights import ROOT,load


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-rows',action='store_true')
    parser.add_argument('--head-groups',action='store_true')
    parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--fp16',action='store_true')
    parser.add_argument('--blas',action='store_true')
    parser.add_argument('--partial-mass',action='store_true')
    parser.add_argument('--mass-warps',type=int,choices=(2,4,8,16),default=8)
    parser.add_argument('--wide',action='store_true')
    parser.add_argument('--recheck',action='store_true')
    parser.add_argument('--mass-stage',type=int,choices=(1,2,3))
    args=parser.parse_args()
    if args.recheck:args.partial_mass=True
    if args.reuse:args.head_groups=True
    full_rows=args.full_rows
    torch.set_num_threads(4)
    root=ROOT/'artifacts/video_runtime_v1'
    data=torch.load(root/'tail_liveness_v1/input.pt',weights_only=True)
    model,*_=load(checkpoint=ROOT/'artifacts/full_coverage_v1/training/C_s17/deployment.pt')
    q=data['q'].cuda()[0].transpose(0,1).reshape(12,72,64,128)
    q=q.index_select(2,torch.linspace(0,63,16,device='cuda').round().long()).flatten(1,2)
    u=q.float().square().reshape(12,1152,8,16).sum(-1)/128
    qn=q.float().norm(dim=-1)/128**.5
    dest=root/'logit_layout_v1'
    dest.mkdir(exist_ok=True)
    rows=[]
    for p in (216,288,360,432):
        k=data['summary']['keys'][:,:p].cuda().contiguous().flatten(1,2).transpose(1,2)
        st=data['summary']['stats'][:,:p].cuda().contiguous()
        bp=st[:,:,1,11].float()
        c=(bp+bp.amax(-1,keepdim=True)).contiguous()
        graphs=[]
        outputs=[]
        variants=(0,64,128,256) if args.mass_stage else (0,128) if args.recheck else ((0,32,64,128) if args.wide else (0,8,16,32)) if args.partial_mass else (0,-1,99) if args.blas else (0,1) if args.fp16 else (12,6,4,3,2,1) if args.head_groups else (0,-8,-16) if full_rows else (0,8,32,64,128)
        for variant in variants:
            tile=0 if args.partial_mass or args.blas or args.head_groups or args.fp16 else max(0,variant)
            cols=p*4
            z=torch.empty(variant if args.reuse else 12,1152,cols if tile==0 else triton.cdiv(cols,tile)*tile,device='cuda',dtype=torch.float16 if args.fp16 and variant else torch.float32)
            norm=torch.empty(12,1152,2,device='cuda')
            out=torch.empty(12,72,p,device='cuda')
            if args.partial_mass and variant:
                nb=triton.cdiv(p,variant)
                val=torch.empty((864,nb,16,variant),device='cuda')
                part=torch.empty((864,nb,16,2),device='cuda')
            own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1).contiguous()
            def run():
                if args.partial_mass and variant:
                    from .risk_partial_mass import _partial,_finish
                    _partial[(864,nb)](q,k,u,qn,st,model[15].theta,c,val,part,72,p,variant,num_warps=args.mass_warps,enable_fp_fusion=False,**(dict(num_stages=args.mass_stage) if args.mass_stage else {}))
                    _finish[(864,nb)](val,part,out,p,variant,triton.next_power_of_2(nb),num_warps=4,enable_fp_fusion=False)
                    if args.recheck:
                        from .risk_recheck import repair
                        return repair(q,k,u,qn,st,model[15].theta,c,own,out)
                    return
                if args.head_groups:
                    for h in range(0,12,variant):
                        hs=slice(h,h+variant)
                        zz=z if args.reuse else z[hs]
                        qk[(18,triton.cdiv(cols,64),variant)](q[hs],k[hs],zz,1152,cols,128,*q.stride(),*k.stride(),
                            BM=64,BF16_INPUTS=True,SCALE=128**-.5,num_warps=4)
                        _normalize[(variant*1152,)](zz,norm[hs],p,triton.next_power_of_2(cols),
                            num_warps=8 if p<=256 else 16,enable_fp_fusion=False)
                        risk[(variant*72,triton.cdiv(p,16))](u[hs],qn[hs],zz,norm[hs],st[hs],model[15].theta[hs],
                            c[hs],out[hs],72,16,p,16,False,True,True,False,True,num_warps=8,enable_fp_fusion=False)
                    return
                if args.blas and variant:
                    from .blas_qk import qk as blas
                    blas(q,k,z,variant)
                    _normalize[(12*1152,)](z,norm,p,triton.next_power_of_2(cols),num_warps=8 if p<=256 else 16,enable_fp_fusion=False)
                elif variant<0:
                    from .full_row_qk import full_rows as fused
                    fused[(72,12)](q,k,z,norm,72,cols,triton.next_power_of_2(cols),num_warps=-variant,num_stages=1)
                else:
                    qk[(18,triton.cdiv(cols,64),12)](q,k,z,1152,cols,128,*q.stride(),*k.stride(),
                    BM=64,BF16_INPUTS=True,SCALE=128**-.5,LOGIT_TILE=tile,num_warps=4)
                    _normalize[(12*1152,)](z,norm,p,triton.next_power_of_2(cols),LOGIT_TILE=tile,
                                         num_warps=8 if p<=256 else 16,enable_fp_fusion=False)
                risk[(864,triton.cdiv(p,16))](u,qn,z,norm,st,model[15].theta,c,out,72,16,p,16,
                    False,True,True,False,True,LOGIT_TILE=tile,num_warps=8,enable_fp_fusion=False)
            graph,count=capture(run)
            if args.recheck and variant:
                print('flagged',p,int(count),flush=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(10):run()
                    torch.cuda.synchronize()
                perf=[(e.key,e.self_device_time_total/10) for e in prof.key_averages() if e.self_device_time_total]
                print(perf,flush=True)
            native=z if tile==0 else z.view(12,72,-1,16,tile).transpose(2,3).reshape(12,1152,-1)[:,:,:cols]
            graphs.append((graph,z,norm,out,val,part) if args.partial_mass and variant else (graph,z,norm,out))
            outputs.append((native,norm,out))
        times=[[] for _ in graphs]
        for r in range(32):
            for j in range(len(graphs)):
                i=(r+j)%len(graphs)
                a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(10):graphs[i][0].replay()
                b.record();b.synchronize()
                if r>=2:times[i].append(a.elapsed_time(b)/10)
        for i,tile in enumerate(variants):
            ms=statistics.median(times[i])
            row=dict(pages=p,tile=tile,ms=ms,speedup=statistics.median(times[0])/ms,
                     equal=[torch.equal(a[-len(b):],b) for a,b in zip(outputs[0],outputs[i])])
            if args.partial_mass and tile:
                row['equal']=[None,None,torch.equal(outputs[0][2],outputs[i][2])]
            if args.fp16 or args.partial_mass:
                from .fixed_page_select import select
                from .selected_id_order import tie_order
                own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1)
                def lut(scores):
                    ids,_=select(scores,own)
                    return ids if p//10>32 else tie_order(scores,ids,p//10)
                ids=lut(outputs[i][2])
                old_ids=lut(outputs[0][2])
                row.update(risk_max_abs=float((outputs[0][2]-outputs[i][2]).abs().max()),
                           changed_sets=int((ids.sort().values!=old_ids.sort().values).any(-1).sum()),
                           changed_order=int((ids!=old_ids).any(-1).sum()))
            print(row,flush=True)
            rows.append(dict(**row,samples_ms=times[i]))
        (dest/(f'mass_stage{args.mass_stage}_w{args.mass_warps}.json' if args.mass_stage else 'recheck.json' if args.recheck else f'partial_mass_wide_w{args.mass_warps}.json' if args.partial_mass and args.wide else 'partial_mass.json' if args.partial_mass else 'blas.json' if args.blas else 'fp16.json' if args.fp16 else 'reused_heads.json' if args.reuse else 'head_groups.json' if args.head_groups else 'full_rows.json' if full_rows else 'module.json')).write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
