"""Full-Q context with mmap raw KV; original random128 queries supervise loss."""
from pathlib import Path
import random
import torch
from ..common import event_key,load_raw
from ..opt11.data import dense

class Events:
    def __init__(self,m,layer):
        self.items=[]
        for row in m['rows']:
            f=Path(m['full_query_cache'])/event_key(row)/f'layer{layer:02d}.pt'
            query=torch.load(f,map_location='cpu',weights_only=True,mmap=True)
            assert query['sample_id']==row['sample_id'] and query['query_reuse_bit_identical']
            assert query['q'].shape==(12,4680,128) and query['indices']==list(range(4680))
            raw=load_raw(m['parent_experiment'],row,layer,device='cpu')
            assert torch.equal(query['q'][:,row['query_indices']],raw['q'])
            raw.pop('q')
            self.items.append(dict(row=row,q=query['q'],raw=raw,truth=None))
        self.train=[x for x in self.items if x['row']['split']=='train']
        self.val=[x for x in self.items if x['row']['split']=='val']
        assert len(self.train)==192 and len(self.val)==96
        assert not {x['row']['prompt_id']for x in self.train}&{x['row']['prompt_id']for x in self.val}

    def gpu(self,item):
        q=item['q'].cuda();raw={k:v.cuda()for k,v in item['raw'].items()}
        targets=torch.tensor(item['row']['query_indices'],device='cuda',dtype=torch.long)
        if item['truth'] is None:
            item['truth']=dense(q[:,targets],raw).float().cpu()
        return q,raw,item['truth'].cuda(),targets

    def calibration_items(self,seed=17,count=12):
        pool=list(self.train);random.Random(seed).shuffle(pool);chosen=[];cells=set()
        for item in pool:
            cell=(item['row']['history_tokens'],item['row']['step'])
            if cell not in cells:chosen.append(item);cells.add(cell)
            if len(chosen)==count:return chosen
        return chosen

    @torch.no_grad()
    def calibration_tensors(self,seed=17,count=12):
        qs=[];ks=[];items=self.calibration_items(seed,count)
        for item in items:
            assert item['row']['split']=='train'
            q=item['q'];raw=item['raw'];k=torch.cat((raw['kh'],raw['kc']),1)
            qi=torch.linspace(0,q.shape[1]-1,256).long();ki=torch.linspace(0,k.shape[1]-1,256).long()
            qs.append(q[:,qi].cuda());ks.append(k[:,ki].cuda())
        return torch.cat(qs,1),torch.cat(ks,1),[x['row']['sample_id']for x in items]
