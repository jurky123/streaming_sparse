"""Frozen numeric semantics. Certified sign interval for exact stored model inputs."""
from ..e42_local_coverage.core import *
import ctypes
D=ROOT/'artifacts/e49_numeric_v1';SO=D/'interval.so';E48=ROOT/'artifacts/e48_tail_state_v1'
def cert(rp,beta,lm,center,mask,r,a,policy):
 lib=ctypes.CDLL(str(SO));ptr=ctypes.POINTER(ctypes.c_double);lib.bounds.argtypes=[ptr,ptr,ptr,ptr,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
 arr=[np.ascontiguousarray(x.detach().cpu().double().numpy()) for x in [rp,beta,lm,center]];mm=np.ascontiguousarray(mask.cpu().numpy(),dtype=np.uint8);out=np.empty(2,np.float64);M,P,K=arr[0].shape;lib.bounds(*[x.ctypes.data_as(ptr) for x in arr],mm.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),M,P,K,int(r),int(a),int(policy),out.ctypes.data_as(ptr));return out
@torch.no_grad()
def paths(rp,beta,lm,center,mask,r,a):
 rp,beta,center=rp.double(),beta.double(),center.double();le=lm+beta;target=(le.softmax(1)[...,None]*(center+rp)).sum(1)
 def output(ms):
  log=torch.where(ms[None],le[:,None],lm[:,None]);w=log.softmax(-1);return (w[...,None]*(center[:,None]+ms[None,:,:,None]*rp[:,None])).sum(-2)
 H=output(mask[None])[:,0];E=target-H;un=[];inc=[];fallback=0;noop=0.
 for st in range(0,len(r),16):
  rr,aa=r[st:st+16],a[st:st+16];ms=mask[None].expand(len(rr),-1).clone();ii=torch.arange(len(rr),device='cuda');ms[ii,rr]=False;ms[ii,aa]=True;HH=output(ms);delta=HH-H[:,None];un.append((2*(E[:,None]*delta).sum(-1)-delta.square().sum(-1)).mean(0))
  logs=torch.where(ms[None],le[:,None],lm[:,None]);sh=logs.max(-1).values;den=(logs-sh[...,None]).exp().sum(-1);h=H[:,None];cr=center[:,rr];ca=center[:,aa];vr=cr+rp[:,rr];va=ca+rp[:,aa]
  expr=(lm[:,rr]-sh).exp()[...,None]*(cr-h)-(le[:,rr]-sh).exp()[...,None]*(vr-h)+(le[:,aa]-sh).exp()[...,None]*(va-h)-(lm[:,aa]-sh).exp()[...,None]*(ca-h);dd=expr/den[...,None];bad=(~torch.isfinite(dd)).any(-1)|((torch.where(mask[None],le,lm).max(-1).values[:,None]-sh)>100);fallback+=int(bad.sum());dd=torch.where(bad[...,None],delta,dd);inc.append((2*(E[:,None]*dd).sum(-1)-dd.square().sum(-1)).mean(0))
 duplicate=output(torch.stack([mask,mask]));noop=float((duplicate[:,1]-duplicate[:,0]).abs().max());return torch.cat(un),torch.cat(inc),noop,fallback

@torch.no_grad()
def run(seed,limit=0,shard=0,shards=1):
 torch.set_num_threads(2);dest=D/(f'pilot_s{seed}' if limit else f's{seed}_part{shard}');dest.mkdir(parents=True,exist_ok=True);data=Data(seed);sources={'E29':E29/f's{seed}/selected.pt','T0':E39/f'T0_s{seed}/epoch1.pt','Full_Pset':E48/f's{seed}/fullset_final.pt'};models={n:model_from(p).eval() for n,p in sources.items()};frozen={n:{k:v.clone() for k,v in m.state_dict().items()} for n,m in models.items()};rows=[];diag=[];near=[];controls=[];start=time.time()
 dump(dest/'protocol.json',dict(seed=seed,training=False,Test=False,models={n:sha(p) for n,p in sources.items()},sign='original highest candidate only; no second candidate; certify outward interval lower>0, otherwise no-op',interval='binary64 outward nextafter operations; MPFR160 directed exp then directed double conversion. Fixed actual Reader output and lm/center binary values. No dense energy in certificate.',units='Set raw gain; Independent one fixed positive common scale, sign equivalent',highprecision_selection='known2 tails for all models, min abs predicted positive by model/split, first positive by event order; no true gain selection',paths='historical, unified FP64, endpoint delta with positive-sum denominator; unsafe endpoint cancellation fallback unified',limitation='intervals conservative; unresolved does not establish exact gain<=0 or real-world safety'))
 for split in ['train','development']:
  metas=json.loads((CACHE/f'{split}_s{seed}/metadata.json').read_text());metas=metas[:limit] if limit else [m for m in metas if m['index']%shards==shard]
  for i,mm in enumerate(metas):
   d=data.load(mm['event']);base={k:mm[k] for k in ['seed','event','split','trajectory','layer','head','chunk','step','tile']}
   for name,model in models.items():
    rp,bp=model(d['s'],d['F']);old,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);raw=old*d['norm'];uni,inc,no,fb=paths(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a']);assert torch.isfinite(uni).all() and torch.isfinite(inc).all();arrays=dict(event=mm['event'],old=raw.cpu().numpy(),unified=uni.cpu().numpy(),increment=inc.cpu().numpy(),true_gain=d['g'].cpu().numpy(),independent=ind.cpu().numpy(),r=d['r'].cpu().numpy(),a=d['a'].cpu().numpy())
    diag.append(dict(**base,model=name,unified_max_difference=float((uni-raw).abs().max()),increment_max_difference=float((inc-raw).abs().max()),no_op_delta=no,increment_fallback_query_actions=fb))
    for policy,score in [('Set',raw),('Independent',ind)]:
     j=int(score.argmax());original=bool(score[j]>0);lo,hi=cert(rp,bp,d['lm'],d['center'],d['mask'],d['r'][j],d['a'][j],policy=='Independent') if original else (-float('inf'),float('inf'));accept=original and lo>0
     for protocol,ex in [('original',original),('interval',accept)]:rows.append(dict(**base,model=name,policy=policy,protocol=protocol,initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=float(d['g'][j]) if ex else 0.,execute=bool(ex),proposal_gain=float(d['g'][j]),hindsight_gain=max(0.,float(d['g'][j])),proposal=j,remove=int(d['r'][j]),add=int(d['a'][j]),raw_score=float(score[j]),interval_low=float(lo),interval_high=float(hi)))
     if policy=='Set':
      for pp,ss in [('unified',uni),('increment',inc)]:
       jj=int(ss.argmax());ex=bool(ss[jj]>0);rows.append(dict(**base,model=name,policy='Set',protocol=pp,initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=float(d['g'][jj]) if ex else 0.,execute=ex,proposal_gain=float(d['g'][jj]),hindsight_gain=max(0.,float(d['g'][jj])),proposal=jj,remove=int(d['r'][jj]),add=int(d['a'][jj]),raw_score=float(ss[jj]),interval_low='',interval_high=''))
      vals=torch.topk(score,2).values;candidate=dict(**base,model=name,j=j,raw=float(score[j]),gap=float(vals[0]-vals[1]),unified_proposal=int(uni.argmax()),increment_proposal=int(inc.argmax()),unified_old_proposal=float(uni[j]),increment_old_proposal=float(inc[j]),low=float(lo),high=float(hi));near.append(candidate)
    np.savez_compressed(dest/f'{split}_{i:04d}_{name}.npz',**arrays)
   if i%96==0:print('RUN',seed,split,i,round(time.time()-start),flush=True)
  table(dest/'decisions.csv',rows);table(dest/'path_checks.csv',diag);table(dest/'proposal_reference.csv',near)
 for n,m in models.items():assert all(torch.equal(v,frozen[n][k]) for k,v in m.state_dict().items())
 data.check_frozen();dump(dest/'complete.json',dict(seed=seed,events=len(diag)//3,training=False,Test=False,weights_unchanged=True,seconds=time.time()-start));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.seed,a.limit,a.shard,a.shards)
