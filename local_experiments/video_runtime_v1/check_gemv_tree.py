"""Locate which input terms merge before each cancellation in the 256-row GEMV."""
import json
import torch
from .weights import ROOT

@torch.no_grad()
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    n=256;g=216
    # Preserve the cuBLAS batched shape, while varying the three terms per output column.
    test=torch.cartesian_prod(torch.arange(1,n),torch.arange(n))
    rows=[]
    v=torch.ones(g,n,1,device='cuda')
    group=torch.arange(g*32,device='cuda')//32
    col=torch.arange(g*32,device='cuda')%32
    for chunk in test.split(g*32):
        i,j=chunk.T.cuda()
        h,c=group[:len(i)],col[:len(i)]
        x=torch.zeros(g,n,32,device='cuda')
        x[h,j,c]=1.
        x[h,0,c]=1.e20
        x[h,i,c]=-1.e20
        rows.extend(torch.bmm(x.transpose(1,2),v).reshape(-1)[:len(i)].cpu().tolist())
    tree=torch.tensor(rows).view(n-1,n)
    nodes=[]
    for i in range(1,n):
        inside=(tree[i-1]==0).nonzero().flatten().tolist()
        nodes.append(dict(i=i,count=len(inside),members=inside))
    for r in nodes[:24]:print(r['i'],r['count'],r['members'][:24])
    (ROOT/'artifacts/video_runtime_v1/group_gemv_order_v1/tree256.json').write_text(json.dumps(nodes))
if __name__=='__main__':main()
