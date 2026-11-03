"""Sparse-only trainable parameters; legacy center/14-feature route removed."""
import torch
from torch import nn
from ..all_sparse_model import SparseLayer

class Layer(nn.Module):
    group_features=SparseLayer.group_features
    def __init__(self,state):
        super().__init__()
        for name in ('bank','group_v','group_gate','pool_k','pool_v','pool_feature','basis'):
            self.register_parameter(name,nn.Parameter(state[name].detach().clone()))
        theta=self.basis.new_tensor([1,1,1,.5]).expand(12,4).clone()
        self.logtheta=nn.Parameter(theta.expm1().log())
    @property
    def theta(self):return torch.nn.functional.softplus(self.logtheta)
    def freeze_stage(self,stage):
        assert stage in ('summary','route','distill')
        for name,p in self.named_parameters():
            p.requires_grad_(name=='logtheta' if stage=='route' else name!='logtheta')
        # Hard sort and hard topk have no invented gradient. Group features train
        # through local pooling; route fits continuous residual labels separately.

class Model(nn.Module):
    def __init__(self,checkpoint,device='cuda'):
        super().__init__();saved=torch.load(checkpoint,map_location=device,weights_only=False)['model']
        self.layers=nn.ModuleDict({str(l):Layer({name.rsplit('.',1)[-1]:v for name,v in saved.items() if name.startswith(f'layers.{l}.')}) for l in range(1,30)})
    def __getitem__(self,l):return self.layers[str(l)]
