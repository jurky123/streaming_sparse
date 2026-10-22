"""Only the specified self-attention Q/K/V/O receive rank-eight adapters."""
import math
import torch
from torch import nn
from .common import tensor_digest


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=8,generator=None):
        super().__init__();self.base=base.requires_grad_(False);self.multiplier=alpha/rank
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5),generator=generator)
    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            update=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.lora_A),self.lora_B)
        return self.base(x)+(self.multiplier*update).to(x.dtype)


def inject(model,layers=(4,14,24),seed=0):
    model.requires_grad_(False)
    generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    for layer in layers:
        for projection in ('q','k','v','o'):
            old=getattr(model.blocks[layer].self_attn,projection)
            if isinstance(old,LoRALinear):raise ValueError('LoRA already installed')
            setattr(model.blocks[layer].self_attn,projection,LoRALinear(old,generator=generator))


def lora_parameters(model):
    return [(n,p) for n,p in model.named_parameters() if n.endswith(('lora_A','lora_B'))]


def base_digest(model):
    # Strip the wrapper's .base name so a teacher and zero-LoRA copy compare.
    return tensor_digest([(n.replace('.base.','.'),p) for n,p in model.named_parameters()
                          if not n.endswith(('lora_A','lora_B'))])


@torch.no_grad()
def effective_update_ratios(model):
    return {n:float((m.multiplier*(m.lora_B@m.lora_A)).norm()/m.base.weight.float().norm().clamp_min(1e-12))
            for n,m in model.named_modules() if isinstance(m,LoRALinear)}


def configure(model,router,group):
    if group not in ('F-K','F-V','A-K','A-V','J-K','J-V'):raise ValueError(group)
    if group[-1]!=router.kind:raise ValueError('Information model/group mismatch')
    model.requires_grad_(False)
    for _,p in lora_parameters(model):p.requires_grad_(group.startswith(('A','J')))
    router.requires_grad_(not group.startswith('A'))
    return dict(group=group,backbone=[dict(name=n,shape=list(p.shape),count=p.numel())
                for n,p in model.named_parameters() if p.requires_grad],
                router=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in router.named_parameters() if p.requires_grad])
