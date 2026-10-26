import argparse,torch,statistics,traceback,math
from .setup import ROOT,tag,CONFIGS
from .native import Native
from .retile import select
from ..opt24.probe import load
from ..opt5.probe import capture,timing
from ..opt23.bench import nmse
from ..common import atomic_json
@torch.no_grad()
def validate(lib,q,ids,ix,actual):
 errors=[];n=q.shape[1]
 for h in [0,5,11]:
  for tile in [0,ids.shape[1]//2,ids.shape[1]-1]:
   qp=q[h,tile*lib.g:min((tile+1)*lib.g,n)].float();scale=qp.abs().max().clamp_min(1.27e-8)/127.;qi=(qp/scale).round();positions=torch.tensor(sorted({0,len(qp)//2,len(qp)-1}),device=q.device);qi=qi[positions]
   pages=ids[h,tile].long();offset=torch.arange(lib.k,device=q.device);valid=pages[:,None]%lib.pages*lib.k+offset<lib.seg
   packed=pages[:,None]*lib.k+offset;kk=lib.ki[h,packed[valid]].float();ks=lib.ks[h,pages][:,None].expand(-1,lib.k)[valid]
   logits=(qi@kk.T)*(scale*ks[None])/(128**.5);expected=logits.softmax(-1).half().float()@lib.v[h,packed[valid]].float()
   oldpos=ix[h,tile*lib.g+positions].long();errors.append(nmse(actual[h,oldpos],expected))
 assert max(errors)<1e-4,errors
 return max(errors)
@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 for k,v in [('height',720),('width',1280),('chunks',4),('layer',24)]:p.add_argument('--'+k,type=int,default=v)
 p.add_argument('--k128-only',action='store_true');p.add_argument('--k64-only',action='store_true');p.add_argument('--confirm',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 raw,model,pipe,ref,aux=load(a.height,a.width,a.chunks,a.layer);q=aux['q_packed'];kv=aux['kv'];ix=aux['q_order'];seg=pipe.seg;b64=kv.state['key'].shape[1];h,n,d=q.shape
 out=ROOT/('confirm' if a.confirm else 'k64_final' if a.k64_only else 'k128' if a.k128_only else 'scan')/tag(a.height,a.width,a.chunks)/f'layer{a.layer:02d}.json';out.parent.mkdir(parents=True,exist_ok=True)
 with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
  dg,truth=capture(lambda:torch.nn.functional.scaled_dot_product_attention(raw['q'][None],kv.k[None],kv.v[None])[0]);dg.replay();truth=truth.clone()
 baseids=select(aux['fine'],aux['candidates'],b64,seg,32,64);baseline=Native(kv,seg,32,64,16);bg,bo=capture(lambda:baseline.prepared(q,baseids,ix));bg.replay();bo=bo.clone();timing(bg,max(5,int(400/max(timing(bg,3),.01))));rows=[]
 for g,k,w in CONFIGS:
  if a.k128_only and k!=128:continue
  if a.k64_only and k!=64:continue
  if a.confirm and (g,k,w) not in [(32,64,16),(64,64,16 if a.height==480 else 32),(128,64,32),(64,128,16)]:continue
  opt=dict(q=g,k=k,warp_q=w,warps=g//w);print('configuration',opt,flush=True)
  try:
   ids=select(aux['fine'],aux['candidates'],b64,seg,g,k);pages=(b64//math.ceil(seg/64))*math.ceil(seg/k);assert bool(((ids>=0)&(ids<pages)).all()) and bool((ids[:,:,1:]>ids[:,:,:-1]).all()),'invalid selection indices';lib=Native(kv,seg,g,k,w);fn=lambda:lib.prepared(q,ids,ix);cg,co=capture(fn);cg.replay();torch.cuda.synchronize();assert torch.isfinite(co).all();err=validate(lib,q,ids,ix,co)
   timing(cg,max(5,int(350/max(timing(cg,3),.01))));samples=[]
   for rep in range(5):
    if rep%2:old=timing(bg,15);dense=timing(dg,10);new=timing(cg,15)
    else:new=timing(cg,15);dense=timing(dg,10);old=timing(bg,15)
    samples.append([old,new,dense])
   old=statistics.median(x[0] for x in samples);new=statistics.median(x[1] for x in samples);dense=statistics.median(x[2] for x in samples)
   qw=(n-torch.arange(ids.shape[1],device=q.device)*g).clamp(max=g).float();length=(seg-(ids%lib.pages)*k).clamp(max=k).float();keep=float((length.sum(-1)*qw).sum()/(h*n*kv.k.shape[1]))
   row=dict(options=opt,ms=new,baseline_ms=old,dense_ms=dense,speedup=old/new,time_vs_dense=new/dense,samples=samples,execution=ids.numel()*g*k/(h*n*kv.k.shape[1]),true_keep=keep,nmse_vs_dense=nmse(co,truth),nmse_vs_q32=nmse(co,bo),reference_max_nmse=err,baseline_nmse_vs_dense=nmse(bo,truth))
   del cg,co,lib
  except Exception as e:
   traceback.print_exc();row=dict(options=opt,error=str(e))
   if 'illegal memory access' in str(e):raise
  rows.append(row);print(row,flush=True);atomic_json(out,dict(completed=False,rows=rows))
 atomic_json(out,dict(completed=True,rows=rows))
if __name__=='__main__':main()
