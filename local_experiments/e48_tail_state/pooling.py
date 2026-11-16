from .run import *
from ..review_v1.ops import deployed
@torch.no_grad()
def run(seed):
 torch.set_num_threads(2);data=Data(seed);rows=[]
 for event in ['data/forest_run_s702/c8_l29_s0_h9_t39.pt','data/reef_fish_s703/c5_l29_s0_h9_t37.pt']:
  d=data.load(event);m=d['meta'];ev=torch.load(OUT/event,map_location='cpu',weights_only=True);sx=torch.load(CACHE/f'train_s{seed}'/m['summary_file'],map_location='cpu',weights_only=True);kv=torch.load(OUT/sx['kv_file'],map_location='cpu',weights_only=True);layer=data.bundle.layers[f'l{m["layer"]}h{m["head"]}'];model=model_from(E29/f's{seed}/selected.pt');rp,bp=model(d['s'],d['F']);sc,_=local_scores(rp,bp,d['lm'],d['center'],d['mask'],d['r'],d['a'],d['norm']);j=int(sc.argmax());newmask=d['mask'].clone();newmask[d['r'][j]]=False;newmask[d['a'][j]]=True
  for p in [int(d['r'][j]),int(d['a'][j])]:
   k=deployed(kv['k'][p:p+1][None].cuda());v=deployed(kv['v'][p:p+1][None].cuda());kg=k.reshape(1,1,4,16,128);vg=v.reshape_as(kg)
   if p>=sx['history_pages']:
    f=torch.einsum('hptd,hrd->hptr',k-k.mean(2,keepdim=True),layer.bank)/128**.5;f=f/f.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
   else:f=layer.group_features(k,v)
   f=f.reshape(1,1,4,16,32);z=torch.einsum('hprtd,hrd->hprt',kg-kg.mean(3,keepdim=True),layer.pool_k)+torch.einsum('hprtd,hrd->hprt',vg-vg.mean(3,keepdim=True),layer.pool_v)+torch.einsum('hprtf,hrf->hprt',f,layer.pool_feature);zz=z-z.mean(-1,keepdim=True)
   for bounded in [True,False]:
    a=(2*torch.tanh(zz/2) if bounded else zz).softmax(-1);pk=(a[...,None]*kg).sum(3)
    for recenter in [True,False]:
     keys=deployed(pk-pk.mean(2,keepdim=True)+k.mean(2,keepdim=True) if recenter else pk).bfloat16();error=float((keys[0,0].float()-d['s'].keys[0,p].float()).abs().max());ss=Summary(*(getattr(d['s'],n).clone() for n in ['keys','mean','detail','stats']));ss.keys[0,p]=keys[0,0];q=ev['q'].cuda().float();c=contributions(q,ss,layer.basis,ev['log_scale'].cuda().double());N,Z,od=[ev[n].cuda().double() for n in ['N','Z','dense']];b=float(objective(c,N,Z,od,d['mask']));aJ=float(objective(c,N,Z,od,newmask));beta=ev['page_logz'].cuda().double()[:,p]-ev['log_scale'].cuda().double()-c['logzh'][:,p]
     rows.append(dict(seed=seed,event=event,page=p,role='remove' if p==int(d['r'][j]) else 'add',bounded=bounded,recenter=recenter,baseline_key_max_error=error,baseline_exact=error==0.,pooling_max_weight=float(a.max()),beta_mean=float(beta.mean()),beta_max=float(beta.max()),initial_nmse=b,after_nmse=aJ,executed_gain=b-aJ,scope='single-page Key intervention; Value detail/Risk/stats frozen; offline only'))
 table(D/f's{seed}/pooling_counterfactual.csv',rows);print('POOL DONE',seed,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
