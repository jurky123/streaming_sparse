"""Post-screen failure diagnosis: all sparse layers and all heads on sailboat.

Dense trajectory remains unchanged. Any derived policy is calibrated ONLY on
sailboat; skateboard/dog are held out for the subsequent selective pilot.
"""
import csv
import json
from pathlib import Path
import torch
from .cache_grouping import history_order,reorder,page_stats
from .screen_cache_research import evaluate


class DiagnosticAttention:
    def __init__(self,config,output,current,original):
        self.config=config;self.output=Path(output);self.current=current;self.original=original
        self.rows=[];self.orders={};self.chunk=-1

    @torch.no_grad()
    def __call__(self,q,k,v,*args,**kwargs):
        c=self.current
        if c['chunk'] in self.config['chunks'] and c['layer']>0 and c['timestep']>0:
            if c['chunk']!=self.chunk:self.orders={};self.chunk=c['chunk']
            nh=k.shape[1]-q.shape[1];h=q.shape[2];d=q.shape[3]
            kh=k[0,:nh].transpose(0,1);vh=v[0,:nh].transpose(0,1)
            kc=k[0,nh:].transpose(0,1);vc=v[0,nh:].transpose(0,1)
            qfull=q[0].transpose(0,1).float();ids=torch.linspace(0,qfull.shape[1]-1,32,device=q.device).long()
            query=qfull[:,ids];route=qfull.reshape(h,-1,64,d).mean(2)[:,ids//64]
            for group,weight in [('k',0.),('kv1',1.)]:
                key=(c['layer'],group)
                if key not in self.orders:self.orders[key]=history_order(kh,vh,weight)
                kk=reorder(kh,self.orders[key]);vv=reorder(vh,self.orders[key]);stats=page_stats(kk,vv)
                for mode in ['mass','var_tail']:
                    metrics,_,_,_,_=evaluate(query,route,kk,vv,kc,vc,stats,mode)
                    for head in range(h):
                        self.rows.append(dict(c,head=head,group=group,mode=mode,
                                              **{key:float(val[head]) for key,val in metrics.items()}))
        return self.original(q,k,v,*args,**kwargs)

    def finish(self,latent,inputs_path):
        old=torch.load(Path(inputs_path).parent/'dense.latents.pt',map_location='cpu',weights_only=True)
        assert torch.equal(old,latent.cpu())
        fields=list(dict.fromkeys(k for r in self.rows for k in r))
        with (self.output/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(self.rows)
        result=dict(config=self.config,rows=len(self.rows),dense_latent_identical=True)
        (self.output/'research.json').write_text(json.dumps(result,indent=2)+'\n')
