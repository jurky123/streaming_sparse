import json,time,torch
from policy import initial,pool,pairs
from scan import counts

t=time.monotonic(); records=[]
for p in (216,288,360,432):
    for idx in (11,21):
        risk=torch.randn(12,72,p,device='cuda',generator=torch.Generator(device='cuda').manual_seed(701))
        own=(torch.arange(72,device='cuda')+p-72).expand(12,-1)
        mask=initial(risk,own,counts(p)[idx])
        r,a,ok=pool(risk,mask,own)
        active=torch.tensor([1,5,10],device='cuda')
        for x,y in zip(pool(risk[active],mask[active],own[active]),(r,a,ok)):
            assert torch.equal(x,y[active])
        assert ok.any((-2,-1)).all()
        for h,tile in ((1,0),(5,33),(10,71)):
            rm,ad=pairs(risk[h,tile],mask[h,tile],own[h,tile])
            moves=mask[h,tile].expand(len(rm),-1).clone()
            rows=torch.arange(len(rm),device='cuda')
            moves[rows,rm]=False; moves[rows,ad]=True
            cur,hist=counts(p)[idx]
            assert moves[:,:24].all() and moves[:,own[h,tile]].all()
            assert (moves[:,-72:].sum(-1)==cur).all() and (moves[:,24:-72].sum(-1)==hist).all()
        records.append([p,idx,int(ok.sum((-1,-2)).min()),int(ok.sum((-1,-2)).max())])
print(json.dumps(dict(seconds=time.monotonic()-t,rows=records)))
