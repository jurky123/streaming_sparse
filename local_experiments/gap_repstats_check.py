import json,torch,triton
from .gap_repstats import stats
from .learned_cache.io import atomic
@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.manual_seed(719);rows=[]
    x=torch.randn(12,72,16,410,4,device='cuda')*4;z=x.logsumexp(-1)+2.772588722239781;w=x.softmax(-1)
    for order in [0,1,2]:
        for div in [0,1,2]:
            for exp in [0,1]:
                a,b=stats(x,order,div,exp);row=dict(order=order,div=div,exp=exp,z_exact=float((a==z).float().mean()),within_exact=float((b==w).float().mean()),within_max=float((b-w).abs().max()))
                rows.append(row);print(json.dumps(row),flush=True)
    best=max(rows,key=lambda x:x['within_exact']);atomic('artifacts/gap_profile_20261031/repstats_checks.json',dict(rows=rows,best=best))
if __name__=='__main__':main()
