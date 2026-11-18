"""Measure staging into the graph and check route LUT, logits, masks and scores."""
import json
import statistics
import torch

from ..review_v1.ops import Summary, Config
from .deployment import current, configure
from .pipeline_graph import BatchGraphs
from .profile_route import capture, inputs
from .risk_projected import risk_inline
from .weights import ROOT


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    configure(current())
    root=ROOT/'artifacts/video_runtime_v1'
    saved=torch.load(root/'fast_attention_baseline_v1/route_capture/route_entry.pt',weights_only=True)
    reference,_,_=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt')
    rows=[]
    for p in (216,432):
        s=Summary(*(saved['summary'][k][:,:p].cuda().contiguous() for k in ('keys','mean','detail','stats')))
        en={k:v[:,:p].cuda().contiguous() for k,v in saved['en'].items()}
        q=saved['q'].cuda().bfloat16()
        if q.size(2)==64:
            q=q.index_select(2,saved['offsets'].cuda())
        theta,basis=(saved[k].cuda() for k in ('theta','basis'))
        own=(torch.arange(72,device='cuda')+p-72)[None].expand(12,-1)
        for correct in (False,True):
            batch=BatchGraphs(risk_inline,reference.phi,Config(),compile_route=True,action_kernel=True,
                              tensor_math=True,compact_reader=True,probes_only=True,bf16_qk=True,input_m16=True,static_route=True)
            batch.native_route_qk=batch.scale_in_qk=True
            batch.reuse_selected_ids='ties'
            query=torch.empty((1,4608,12,128),device=q.device,dtype=q.dtype)
            query=query[0].transpose(0,1).reshape(12,72,64,128)
            offsets=saved['offsets'].cuda()
            query.index_copy_(2,offsets,q)
            e=batch.get(q,s,en,theta,basis,own,correct)
            baseline=[x.clone() for x in batch.replay(e,q,own)]
            # Poison the unused graph inputs before the candidate replay.
            for tensor in (e['s'].mean,e['s'].detail,e['basis']):
                tensor.fill_(float('nan'))
            batch.projected_route_inputs=True
            probes=batch.select_probes(query,offsets)
            assert probes is batch.probe_buffer and torch.equal(probes,q)
            batch.entries.clear()
            e=batch.get(probes,s,en,theta,basis,own,correct)
            assert e['q'] is probes
            for tensor in (e['s'].mean,e['s'].detail,e['basis']):
                tensor.fill_(float('nan'))
            batch.stage(e,s,en,theta,basis)
            new=batch.replay(e,probes,own)
            eq=[torch.equal(x,y) for x,y in zip(baseline,new)]
            variants=[]
            for enabled in (False,True):
                batch.projected_route_inputs=enabled
                def stage_inputs():
                    selected=batch.select_probes(query,offsets)
                    if selected is not e['q']:e['q'].copy_(selected)
                    batch.stage(e,s,en,theta,basis)
                variants.append(capture(stage_inputs)[0])
            times=[[],[]]
            for rep in range(24):
                for i in (rep%2,1-rep%2):
                    a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                    a.record()
                    for _ in range(16):variants[i].replay()
                    b.record();b.synchronize()
                    if rep>=4:times[i].append(a.elapsed_time(b)/16)
            ms=list(map(statistics.median,times))
            row=dict(pages=p,correct=correct,equal=eq,ms=ms,speedup=ms[0]/ms[1])
            print(row,flush=True)
            rows.append(dict(**row,times_ms=times))
    path=root/'projected_route_inputs_v1'
    path.mkdir(exist_ok=True)
    (path/'module.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
