from ..e26_oracle_coefficients.core import *
E27=ROOT/'artifacts/e27_adaptive_query_v1'

def page_features(s):
 parts=[s.keys[0].float().flatten(1),s.mean[0].float(),s.detail[0].float().flatten(1),s.stats[0].float().flatten(1)]
 return torch.cat([p/(p.detach().square().mean(-1,keepdim=True).sqrt()+1e-4) for p in parts],1)

class AdaptivePhi(nn.Module):
 def __init__(self):
  super().__init__();self.page=nn.Sequential(nn.Linear(712,64),nn.GELU(),nn.Linear(64,32));self.query=nn.Linear(128,8,bias=False)
  nn.init.orthogonal_(self.query.weight);self.query.weight.data/=math.sqrt(128)
 def forward(self,q,s):
  A=self.page(page_features(s)).reshape(-1,8,4).bfloat16().float();u=self.query(q.float());phi=torch.tanh(torch.einsum('qi,pij->qpj',u,A)).double()
  return torch.cat([torch.ones_like(phi[...,:1]),phi],-1)

def ls_response(F,lm,target,zscale,fit_queries=None):
 # Inner ridge is identical to E26; optional fit16, predict64 is diagnostic.
 maxlm=lm.max(0).values;X=((lm-maxlm[None]).exp()[...,None]*F).permute(1,0,2);Y=target.permute(1,0,2)
 if fit_queries is None:xf,yf=X,Y
 else:xf,yf=X[:,fit_queries],Y[:,fit_queries]
 coef,_,_=ridge_fit(xf,yf,zscale)
 return (X@coef).permute(1,0,2)

def variable_projection_loss(F,lm,target,zscale):
 X=((lm-lm.max(0).values[None]).exp()[...,None]*F).permute(1,0,2);Y=target.permute(1,0,2)
 # Purely numerical target scaling; normalized outer weighting is explicit.
 scale=Y.abs().amax((1,2)).clamp_min(zscale).detach();Yn=Y/scale[:,None,None]
 G=X.transpose(1,2)@X;reg=RIDGE*G.diagonal(dim1=-2,dim2=-1).sum(-1)/5
 coef=torch.linalg.solve(G+reg[:,None,None]*torch.eye(5,dtype=G.dtype,device=G.device),X.transpose(1,2)@Yn)
 err=(X@coef-Yn).square().sum((1,2));energy=Yn.square().sum((1,2))+64*16*(zscale/scale).square()
 loss=(err/energy).mean()
 return loss

def prepare(bundle,path):
 with torch.no_grad():
  d=load(path);s=bundle.make(d,'C');c=contributions(d['q'],s,bundle.layers[d['key']].basis,d['log_scale']);fit=torch.load(OUT/'fit.pt',weights_only=True);B=fit['B'].cuda();zt,_,bt,_=labels(c,d['N'],d['Z'],d['dense'],B,d['logZ']);lm=c['logzh']-c['A'].log()[:,None]
 return d,s,c,zt,bt,lm

@torch.no_grad()
def evaluate(net,bundle,seed,paths):
 qs=torch.linspace(0,63,16,device='cuda').round().long();fit=torch.load(OUT/'fit.pt',weights_only=True);rows=[]
 for idx,path in enumerate(paths):
  d,s,c,zt,bt,lm=prepare(bundle,path);F=net(d['q'],s);z=ls_response(F,lm,zt,fit['zscale']);mask,r,a=route(d['q'][qs],s,bundle.layers[d['key']],d['own']);g=gains(c,d['N'],d['Z'],d['dense'],mask,r,a);J=float(objective(c,d['N'],d['Z'],d['dense'],mask));norm=d['dense'][qs].square().sum()/16+1e-12;score,ind=safe_predictions(z[qs],bt[qs],lm[qs],mask,r,a,norm)
  row=dict(event=str(path.relative_to(OUT)),trajectory=d['ev']['trajectory'],seed=seed,initial=J,oracle_gain=max(0.,float(g.max())),z_nmse=nmse(z,zt),unselected_z_nmse=nmse(z[:,~mask],zt[:,~mask]),reconstruction_loss=float(variable_projection_loss(F,lm,zt,fit['zscale'])))
  for name,v in [('set',score),('independent',ind)]:
   j=int(v.argmax());ex=bool(v[j]>0);row.update({name+'_execute':ex,name+'_gain':float(g[j]) if ex else 0.,name+'_proposal_gain':float(g[j]),name+'_remove':int(r[j]),name+'_add':int(a[j])})
  rows.append(row)
 return rows

# Stable gain identity: ||E||²-||E-deltaO||² = 2E·deltaO-||deltaO||².
# Avoid subtracting nearly equal very large objective values.
def safe_predictions(z,beta,logmass,mask,r,a,normalizer):
 le=logmass+beta.double();shift=torch.maximum(logmass.max(1).values,le.max(1).values);shift=torch.maximum(shift,z.abs().amax((1,2)).clamp_min(1e-300).log());mass=(logmass-shift[:,None]).exp();exact=(le-shift[:,None]).exp();zz=z*torch.exp(-shift)[:,None,None]
 small=beta.abs()<.5;smallbeta=torch.where(small,beta,torch.zeros_like(beta));u=torch.where(small,mass*torch.expm1(smallbeta.double()),exact-mass)
 Da=exact.sum(1);Ds=torch.where(mask[None],exact,mass).sum(1);Ts=zz[:,mask].sum(1);H=Ts/Ds[:,None]
 E=(zz[:,~mask].sum(1)-H*u[:,~mask].sum(1)[:,None])/Da[:,None];scores=[]
 for st in range(0,len(r),32):
  rr,aa=r[st:st+32],a[st:st+32];ms=mask[None].expand(len(rr),-1).clone();idx=torch.arange(len(rr),device=z.device);ms[idx,rr]=False;ms[idx,aa]=True;mf=ms.double();den=exact@mf.T+mass@(1-mf).T
  if not bool((den>0).all()):raise FloatingPointError('nonpositive predicted denominator')
  delta=(zz[:,aa]-zz[:,rr]-H[:,None]*(u[:,aa]-u[:,rr])[...,None])/den[...,None]
  scores.append((2*(E[:,None]*delta).sum(-1)-delta.square().sum(-1)).mean(0)/normalizer)
 score=torch.cat(scores);target=zz.sum(1)/Da[:,None];net=zz-target[:,None]*u[...,None];energy=(net/Da[:,None,None]).square().sum(-1).mean(0);ind=energy[a]-energy[r]
 if not bool(torch.isfinite(score).all() and torch.isfinite(ind).all()):raise FloatingPointError('nonfinite predicted score')
 return score,ind
