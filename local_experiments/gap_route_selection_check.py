"""Matched-input native/tensor-core final page selection audit, no QKV dumps."""
import json,os,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_route_tensor_runtime import GapRouteTensorRuntime
from .gap_input_runtime import GapInputRuntime
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,disk_guard,RunLock
from .gap_group_isolated import device_peers
@torch.no_grad()
def main():
    root=Path('artifacts/gap_broad_opt_20261031/route_selection');lock=RunLock(root);rt=None
    try:
        assert not device_peers();disk_guard(root,50,1);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        engine=Engine();rt=GapRouteTensorRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load('artifacts/four_gaps_20261031/shared/epoch1.pt',map_location='cpu',weights_only=False)['model'])
        candidate=rt.decision;rows={}
        def audit(l,q,pk,*args):
            native=GapInputRuntime.decision(rt,l,q,pk,*args)[0].clone()
            got=candidate(l,q,pk,*args);chosen=got[0]
            a=native.sort(-1).values;b=chosen.sort(-1).values
            diff=(a!=b).any(-1);missing=(~(a[...,None]==b[...,None,:]).any(-1)).sum()
            row=rows.setdefault(f'{l}:{pk.shape[1]}',dict(layer=l,pages=pk.shape[1],calls=0,query_head_tiles=0,changed_tiles=0,selected_slots=0,replaced_pages=0))
            row['calls']+=1;row['query_head_tiles']+=diff.numel();row['changed_tiles']+=int(diff.sum());row['selected_slots']+=a.numel();row['replaced_pages']+=int(missing)
            return got
        rt.decision=audit
        emb=torch.load(DATA/'embeddings/horse_field.pt',weights_only=True).cuda()
        for c in range(7):
            generate_chunk(engine,emb,101,c)
            atomic(root/'checks.json',dict(status='running',chunk=c,rows=list(rows.values())))
        assert not device_peers()
        sums={k:sum(x[k] for x in rows.values()) for k in ['calls','query_head_tiles','changed_tiles','selected_slots','replaced_pages']}
        atomic(root/'checks.json',dict(status='complete',totals=sums,rows=list(rows.values()),scope='Matched input final selected page sets across all sparse layers in seven chunks; trajectory follows candidate. Not video quality acceptance.'))
        print(json.dumps(sums),flush=True)
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
