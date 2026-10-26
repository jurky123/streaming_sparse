import argparse
from pathlib import Path
import torch
from ..common import atomic_json
from ..opt2.attention import attention as reference_attention
from ..opt5.probe import capture,timing
from .split_attention import attention

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True)
    for k,v in dict(group=32,splits=4,warps=4,stages=3,n=4680).items():p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--cache',default='');a=p.parse_args();root=Path(a.directory);out=root/'optimization7/split_probes';out.mkdir(parents=True,exist_ok=True)
    opts={k:getattr(a,k)for k in ['group','splits','warps','stages']};name='_'.join(f'{k}{v}'for k,v in opts.items())+f'_n{a.n}'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(root/'optimization/stage3_fused_K32/actual_block.pt',map_location='cuda',weights_only=False)
    q=data['q'][:,:a.n];k=data['k'];v=data['v'];mask=data['mask'][:,:a.n]
    ref=lambda:reference_attention(q,k,v,mask,group=32,warps=4,stages=3)
    fn=lambda:attention(q,k,v,mask,**opts)
    yd=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]
    yr=ref();y,counts=attention(q,k,v,mask,**opts,return_union=True);assert torch.isfinite(y).all()
    # Reference is the same masked attention, not full Dense.
    rg,_=capture(ref);ng,_=capture(fn);dg,_=capture(lambda:torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None]))
    for _ in range(4):timing(rg,100);timing(ng,100)
    rows=[]
    for i in range(6):
        pairs=[('reference',rg),('new',ng),('dense',dg)]
        if i%2:pairs.reverse()
        rows.append({s:timing(g,50)for s,g in pairs})
    nmse=lambda x,y:float((x.float()-y.float()).square().sum()/y.float().square().sum())
    result=dict(options=opts,shape=list(q.shape),masked_nmse=nmse(y,yr),reference_to_dense_nmse=nmse(yr,yd),new_to_dense_nmse=nmse(y,yd),max_abs=float((y.float()-yr.float()).abs().max()),union_fraction=float(counts.float().mean()/mask.shape[-1]),
        timings_ms={k:sum(r[k]for r in rows)/len(rows)for k in rows[0]},samples=rows)
    atomic_json(out/(name+'.json'),result);print(result,flush=True)

if __name__=='__main__':main()
