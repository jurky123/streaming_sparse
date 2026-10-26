import math,torch,triton as tr
from ..opt26.route import coarse,refine,select,original_fine,original_select,radix_top,_top
def projected(q,state,seg,mg=8,width=128,coarse_warps=4,gp=8,fine_warps=4,trans=False,radix=False,tile_radix=None,legacy_fine=True):
 h,n,_=q.shape;g=tr.cdiv(n,32);b=state['key'].shape[1];c=math.ceil(b*.35);qq=q
 score=coarse(qq,state,mg,width,coarse_warps)
 if radix:candidates=radix_top(score,c,8,4)
 else:
  candidates=torch.empty((h,g,c),device=q.device,dtype=torch.int32)
  _top[(h*g,)](score,candidates,h*g,b,c,tr.next_power_of_2(b),num_warps=4)
 if legacy_fine:
  fine=torch.empty((h,n,c),device=q.device)
  original_fine[(g,tr.cdiv(c,8),h)](qq,state['key'],state['logcount'],candidates,fine,n,b,c,g,tr.cdiv(seg,64),num_warps=4,num_stages=1)
 else:fine=refine(qq,state,candidates,seg,gp,fine_warps,trans)
 use_radix=radix if tile_radix is None else tile_radix
 if legacy_fine and not use_radix:return original_select(fine,candidates,b,32,.2)
 return select(fine,candidates,n,b,trans,radix=use_radix)


def route(q,state,seg):
 b=state["key"].shape[1];c=math.ceil(b*.35)
 return projected(q,state,seg,mg=8,width=128 if b>1024 else 64,radix=b>512,tile_radix=c>512)
