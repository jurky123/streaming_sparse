from collections import defaultdict
from pathlib import Path
import torch
from ..capture.dataset_io import AttentionDataset


class SnapshotData:
    def __init__(self,root,split):
        self.dataset=AttentionDataset(root,split)
        self.root=Path(root)
        self.groups=defaultdict(list)
        self.strata=defaultdict(list)
        for r in self.dataset.records:
            self.groups[r['cache_version']].append(r)
        for version,records in self.groups.items():
            r=records[0]
            self.strata[(r['layer_id'],r['head_id'])].append(version)
        if not self.groups:
            raise ValueError(f'Empty split {split}')

    def load(self,version,device):
        records=sorted(self.groups[version],key=lambda r:r['step_id'])
        history=torch.load(self.root/records[0]['history'],weights_only=True,map_location='cpu')
        history={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in history.items()}
        events=[]
        for r in records:
            event=torch.load(self.root/r['event'],weights_only=True,map_location='cpu')
            if event['cache_version']!=history['cache_version']:
                raise ValueError('Cache version mismatch')
            events.append({**history,**{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in event.items()}})
        return events

    def sample_versions(self,rng,count):
        keys=list(self.strata)
        rng.shuffle(keys)
        chosen=[]
        for i in range(count):
            key=keys[i%len(keys)]
            # Layer/head is uniform; within each, balance prompt, seed and history.
            chosen.append(rng.choice(self.strata[key]))
        return chosen
