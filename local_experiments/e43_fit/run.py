from ..e42_local_coverage.core import *
D=ROOT/'artifacts/e43_fit_v1'
def run(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);data=Data(seed);metas=list(data.meta.values());trajectories=sorted({m['trajectory'] for m in metas if m['split']=='train'});chosen=[]
 for i,t in enumerate(trajectories):
  for layer in [1,15,29]:
   z=sorted([m for m in metas if m['split']=='train' and m['trajectory']==t and m['layer']==layer and m['chunk']==5 and m['step']==0 and m['head']==[0,3,6,9][i%4]],key=lambda x:x['tile']);assert len(z)==2;chosen+=z
 dest=D/f's{seed}';dest.mkdir(parents=True,exist_ok=True);dump(dest/'protocol.json',dict(events=[m['event'] for m in chosen],updates=2304,states=96,loss='original E29',test=False,selection='trajectory rotating head, chunk5 step0 layers1/15/29 two tiles',final_checkpoint=True))
 states=[];identity={};initial=[];model=model_from(E29/f's{seed}/selected.pt');
 for m in chosen:
  d=data.load(m['event']);sx=torch.load(CACHE/f'train_s{seed}'/m['summary_file'],map_location='cpu',weights_only=True);ids=[]
  for p,h in enumerate(sx['summary_hashes']):
   key=(m['trajectory'],m['layer'],m['head'],h)
   if key not in identity:identity[key]=len(initial);initial.append(model.page(__import__('local_experiments.e29_local_reader.core',fromlist=['page_features']).page_features(d['s']))[p].detach())
   ids.append(identity[key])
  d['ids']=torch.tensor(ids,device='cuda');states.append(d);bg=d.copy();C=torch.zeros_like(d['mask']);C[d['r']]=True;C[d['a']]=True;rr=torch.where(d['mask']&~C)[0];aa=torch.where(~d['mask']&~C)[0];rr=rr[rr!=d['own']];assert len(rr) and len(aa);mask=d['mask'].clone();mask[rr[0]]=False;mask[aa[0]]=True
  st=dict(mask=mask.cpu().tolist(),remove=d['r'].cpu().tolist(),add=d['a'].cpu().tolist());bg=data.load(m['event'],st);bg['ids']=d['ids'];states.append(bg)
 class Lookup(torch.nn.Module):
  def __init__(self):super().__init__();self.co=torch.nn.Parameter(torch.stack(initial))
  def forward(self,d):
   co=self.co[d['ids']].reshape(-1,5,17).bfloat16().float();v=co[None,:,0]+torch.einsum('qpj,pjk->qpk',d['F'][...,1:].float(),co[:,1:]);return v[...,:16],v[...,16]
 lookup=Lookup().cuda();models={'P':model,'L':lookup};opts={k:torch.optim.Adam(v.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8) for k,v in models.items()};curves=[]
 def outputs(k,d):return models[k](d) if k=='L' else models[k](d['s'],d['F'])
 def evaluate(epoch):
  rows=[]
  with torch.no_grad():
   for i,d in enumerate(states):
    for k in ['P','L','Teacher']:
     rp,bp=(d['rtrue'],d['btrue']) if k=='Teacher' else outputs(k,d);sc,ind=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm'])
     for policy,scores in [('Set',sc),('Independent',ind)]:
      j=int(scores.argmax());ex=bool(scores[j]>0);rows.append(dict(epoch=epoch,arm=k,state=i,event=d['meta']['event'],policy=policy,initial=d['initial'],oracle_gain=max(0.,float(d['g'].max())),executed_gain=float(d['g'][j]) if ex else 0.,execute=ex,proposal_gain=float(d['g'][j]),hindsight_gain=max(0.,float(d['g'][j])),local_mse=float((rp-d['rtrue']).square().mean()),beta_mae=float((bp-d['btrue']).abs().mean())))
  table(dest/f'evaluation_{epoch}.csv',rows)
 evaluate(-1)
 for update in range(2304):
  d=states[update%96]
  for k in ['P','L']:
   rp,bp=outputs(k,d);ls,_,_,_,_=losses(rp,bp,d['rtrue'],d['btrue'],d['mass'],d['lm'],d['center'],d['mask'],d['r'],d['a'],d['g'],d['norm'],data.scale,data.fit);loss=sum(ls);opt=opts[k];opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(models[k].parameters(),1,error_if_nonfinite=True);opt.step();curves.append(dict(update=update,arm=k,loss=float(loss),gradnorm=float(gn)))
  if update%384==0:print(seed,update,flush=True)
 evaluate(1);table(dest/'curves.csv',curves)
 for k,m in models.items():torch.save(m.state_dict(),dest/f'{k}_final.pt')
 data.check_frozen();dump(dest/'complete.json',dict(seed=seed,updates=2304,unique_pages=len(initial),P_parameters=sum(p.numel() for p in model.parameters()),L_parameters=lookup.co.numel(),test=False,raw_KV=False));print('DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
