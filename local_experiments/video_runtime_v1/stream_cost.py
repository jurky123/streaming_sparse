"""Logical versus decomposed matmul work for the streamed implementation."""
import json
from pathlib import Path
from .theoretical_cost import H,Q,D,common,attention

def costs(p):
    raw=attention(int(p*.1));rep_qk=2*H*Q*p*4*D;probe_qk=rep_qk/4
    return dict(logical_exact=raw,logical_rep_qk=rep_qk,logical_rep_av=rep_qk,logical_probe_qk=probe_qk,
                logical_total=raw+2*rep_qk+probe_qk,
                issued_tf32x3=raw*1.5+rep_qk+rep_qk*3+probe_qk*3,
                issued_bf16x4=raw*1.5+rep_qk+rep_qk*4+probe_qk)

def generation(chunks):
    dense=sparse=0
    for c in range(chunks):
        p=min(c+1,6)*72;d=attention(p);s=costs(p)['logical_total']+4.5e9
        dense+=5*30*(common+d)
        sparse+=5*30*(common+d) if c<2 else 5*30*common+34*d+116*s
    return dense/sparse

if __name__=='__main__':
    v=costs(432)
    result=dict(full_window_gflop={k:x/1e9 for k,x in v.items()},dense_gflop=attention(432)/1e9,
                logical_attention_speedup=attention(432)/v['logical_total'],with_route_4_to_5g=[attention(432)/(v['logical_total']+g*1e9) for g in (5,4)],
                generation_logical_ratio={str(n):generation(n) for n in (9,21)},
                notes='FMA=2; issued counts include two BF16 exact AV products, three TF32 or four BF16 tail AV products. Issued FLOPs of different datatypes do not predict equal latency. Excludes summary/grouping/encoding/reconstruction/memory/launches/nonlinear operations; extra route4-5G remains approximate.')
    path=Path('artifacts/video_runtime_v1/hotspot_round6/stream_cost.json');path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
