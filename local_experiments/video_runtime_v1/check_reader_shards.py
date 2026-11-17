"""Test register occupancy versus additional shard-merge work, same 72-tile input."""
import json
from pathlib import Path
import torch
from .profile_route import inputs,capture
from .weights import ROOT
from . import reader_reduce as module,scores_moments
from .reader_late import reader_kernel
from .scores import candidate_pool
from ..review_v1.ops import RouteOps,Config
from .benchmark import measure

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    module.AGG32=True;module.WARPS=4;module.SHARD=128;module.MOMENTS=True;module.PACKED=True;module.reader_kernel=reader_kernel;scores_moments.BLOCKED=True
    rt,e,identity=inputs(True,True,reader_checkpoint=ROOT/'artifacts/full_coverage_v1/training/Full_Pset_s17/selected.pt');out=rt.compute(e,True)
    expand=lambda x:x[:,:1].expand(-1,72,*x.shape[2:]).contiguous()
    z=expand(out[1].index_select(2,e['offsets']));q=expand(e['q'].index_select(2,e['offsets']));risk=expand(out[3]);own=expand(e['own']);u=rt.phi.query(q)
    lm=z.double().logsumexp(-1);lm=lm-lm.logsumexp(-1,keepdim=True)
    mask=RouteOps.select(risk,own,Config(),check_valid=False)['mask'];r,a=candidate_pool(risk,mask,own)
    fn=torch.compile(lambda:module.compact_scores(u,e['en'],z,lm,mask,r,a),fullgraph=True,options={'triton.cudagraphs':False})
    rows=[];ref=None
    for shard,warps in [(128,4),(64,4),(32,4),(128,8)]:
        module.SHARD=shard;module.WARPS=warps;g,y=capture(fn)
        if ref is None:ref=y.clone()
        row=dict(shard=shard,warps=warps,timing=measure(g.replay,25),max_abs=float((ref-y).abs().max()),score_nmse=float((ref-y).square().sum()/ref.square().sum()),argmax_mismatches=int((ref.flatten(1).argmax(1)!=y.flatten(1).argmax(1)).sum()),noop_mismatches=int(((ref.flatten(1).max(1).values>0)!=(y.flatten(1).max(1).values>0)).sum()),kernel_stats=module.KERNEL_STATS[(72,z.shape[-2])]);rows.append(row);print(row,flush=True)
    p=ROOT/'artifacts/video_runtime_v1/hotspot_round8/reader_shards.json';p.write_text(json.dumps(dict(rows=rows,identity=identity),indent=2))
if __name__=='__main__':run()
