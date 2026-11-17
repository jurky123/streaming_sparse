"""Dominant-FLOP envelope, not a hardware latency prediction. No GPU required."""
import json
from pathlib import Path

H,Q,D,PAGE,R,RANK = 12,4608,128,64,4,8
WIDTH,FFN,TEXT = 1536,8960,512

def attention(p):
    return 4*H*Q*(p*PAGE)*D

def sparse_main(p):
    return dict(exact=attention(int(p*.1)),
                representative_qk=2*H*Q*p*R*D,
                full_mean=2*H*Q*p*D,
                detail_weight=2*H*Q*p*R*(R-1),
                detail_av=2*H*Q*p*(R-1)*RANK,
                detail_reconstruction=2*H*Q*RANK*D)

# Self Q/K/V/O, two FFN linears, cross Q/O and cross QK/AV.
# Cached cross K/V projections, normalization, modulation, nonlinearities,
# sparse summary/grouping/encoding, memory traffic and launches are omitted.
common = 8*Q*WIDTH**2 + 4*Q*WIDTH*FFN + 4*Q*WIDTH**2 + 4*Q*TEXT*WIDTH

def generation(chunks,route_g):
    dense=sparse=0
    for chunk in range(chunks):
        p=min(chunk+1,6)*72
        d=attention(p)
        s=sum(sparse_main(p).values())+route_g*1e9
        dense+=5*30*(common+d)
        # First two chunks, layer 0 and each clean commit remain Dense.
        sparse+=5*30*(common+d) if chunk<2 else 5*30*common+34*d+116*s
    return dense/sparse

def run():
    p=432;parts=sparse_main(p);d=attention(p)
    result=dict(assumptions='Dominant FLOPs, FMA=2. Equal effective throughput across precisions assumed. Extra routing modeled as 4-5 GFLOP/layer, a planning estimate, not an exact operation census. Omitted summary/commit and memory costs make estimates optimistic.',
                saturated_pages=p,dense_attention_gflop=d/1e9,
                sparse_main_gflop={k:v/1e9 for k,v in parts.items()},
                main_attention_speedup=d/sum(parts.values()),
                attention_with_route_speedup=[d/(sum(parts.values())+g*1e9) for g in (5,4)],
                unchanged_per_layer_gflop=common/1e9,
                generation_dominant_flop_speedup={str(n):[generation(n,g) for g in (5,4)] for n in (9,21)})
    path=Path('artifacts/video_runtime_v1/formal_short_15/theoretical_cost.json')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
