"""Independent selected-raw / unselected-summary FP32 path for small pages."""
import math,json,numpy as np,torch
from pathlib import Path
from ..review_v1.ops import haar
@torch.no_grad()
def check(folder,ti,data,result):
 torch.set_num_threads(1);stored=torch.load(folder/'summary.pt',weights_only=True);snap=torch.load(stored['source_snapshot'],weights_only=True);h=stored['head'];size=int(data['page_size']);offset=int(data['query_offset']);q=snap['q'][h,offset:offset+size].float();k=stored.get('raw_k',snap['k'][h]).reshape(-1,size,128);v=stored.get('raw_v',snap['v'][h]).reshape_as(k);s=stored['summary'];basis=stored['basis'].bfloat16().float();P=k.shape[0];g=size/4;rows={}
 opt=np.zeros(P,bool);opt[result['mask']]=True
 for method,mask in [('mass',data['mass']),('risk',data['risk']),('opt',opt)]:
  ids=torch.from_numpy(np.where(mask)[0]);kk=k[ids].flatten(0,1).float();vv=v[ids].flatten(0,1).float();z=q@kk.T/math.sqrt(128);ze=z.logsumexp(-1);exact=z.softmax(-1)@vv
  zs=(q@s['keys'].flatten(0,1).float().T/math.sqrt(128)).reshape(size,P,4);zs=zs.masked_fill(torch.from_numpy(mask)[None,:,None],-torch.inf);zt=zs.flatten(1).logsumexp(-1)+math.log(g);weights=(zs-(zt-math.log(g))[:,None,None]).exp();mean=weights.sum(-1)@s['mean'].float();contrast=torch.einsum('qpr,ar->qpa',weights,haar(q));code=torch.einsum('qpa,pak->qk',contrast,s['detail'].float());tail=mean+code@basis.T;total=torch.logaddexp(ze,zt);out=(exact*(ze-total).exp()[:,None]+tail*(zt-total).exp()[:,None]).bfloat16().double().numpy()
  target=np.where(mask[None,:,None],data['N'],data['Nh']).sum(1)/np.where(mask[None,:],data['Z'],data['Zh']).sum(1)[:,None];nmse=float(np.square(out-target).sum()/(np.square(target).sum()+1e-12));rows[method]=dict(fp32_vs_fp64_nmse=nmse,max_abs=float(np.abs(out-target).max()),finite=bool(np.isfinite(out).all()),path='FP32 selected raw attention and unselected summaries; BF16 output; generalized page size')
  assert np.isfinite(out).all()
 (folder/f't{ti}'/'fp32_check.json').write_text(json.dumps(rows,indent=2));return rows
