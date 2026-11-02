"""Capture real native Self-Forcing attention operands without changing outputs."""
import json
from pathlib import Path
import torch


class ResearchAttention:
    def __new__(cls,config,output,current,original):
        if config['mode']=='diagnose':
            from .diagnose_cache_layers import DiagnosticAttention
            return DiagnosticAttention(config,output,current,original)
        if config['mode']=='prototype':
            from .cache_sparse_prototype import PrototypeAttention
            return PrototypeAttention(config,output,current,original)
        return super().__new__(cls)

    def __init__(self, config, output, current, original):
        self.config=config;self.output=Path(output);self.current=current;self.original=original
        self.folder=self.output/'qkv';self.folder.mkdir(exist_ok=True)
        self.previous={};self.records=[];self.heads=config.get('heads',[0,4,8])

    @torch.no_grad()
    def __call__(self,q,k,v,*args,**kwargs):
        if self.config['mode']!='capture':
            raise ValueError(self.config['mode'])
        c=self.current
        if c['chunk'] in self.config['chunks'] and c['layer'] in self.config['layers']:
            name=f"c{c['chunk']:02d}_l{c['layer']:02d}"
            nh=k.shape[1]-q.shape[1]
            # CPU copies are diagnostic only, not included in speed measurements.
            kh=k[0,:nh,self.heads,:].transpose(0,1).contiguous().cpu()
            vh=v[0,:nh,self.heads,:].transpose(0,1).contiguous().cpu()
            row=dict(c,history_tokens=nh,current_tokens=q.shape[1],heads=self.heads)
            if name not in self.previous:
                torch.save(dict(k=kh,v=vh,**row),self.folder/f'{name}_history.pt')
                self.previous[name]=(kh,vh,0)
            else:
                oldk,oldv,step=self.previous[name]
                row['history_unchanged']=torch.equal(kh,oldk) and torch.equal(vh,oldv)
                assert row['history_unchanged'],row
                self.previous[name]=(oldk,oldv,step+1)
            step=self.previous[name][2]
            if c['timestep']>0:
                torch.save(dict(q=q[0,:,self.heads,:].transpose(0,1).contiguous().cpu(),
                    k=k[0,nh:,self.heads,:].transpose(0,1).contiguous().cpu(),
                    v=v[0,nh:,self.heads,:].transpose(0,1).contiguous().cpu(),**row),
                    self.folder/f'{name}_s{step}.pt')
            self.records.append(row)
        return self.original(q,k,v,*args,**kwargs)

    def finish(self,latent,inputs_path):
        reference=Path(inputs_path).parent/'dense.latents.pt'
        old=torch.load(reference,map_location='cpu',weights_only=True)
        result=dict(config=self.config,records=self.records,
                    unchanged_history_checks=sum('history_unchanged' in r for r in self.records),
                    identical_to_previous_dense=torch.equal(old,latent.cpu()),
                    max_latent_difference=float((old-latent.cpu()).abs().max()))
        (self.output/'research.json').write_text(json.dumps(result,indent=2)+'\n')
        assert result['identical_to_previous_dense'],result
