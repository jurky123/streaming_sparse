"""Functional history: graph ownership lasts one loss, never an optimizer step."""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class CacheIdentity:
    model_version: str
    router_version: str
    trajectory_id: str
    sample_id: str
    history_start: int
    history_end: int
    context_noise: float
    partition: str = 'contiguous_page64'
    owner: str = 'student'


@dataclass(frozen=True)
class LayerKV:
    k: torch.Tensor
    v: torch.Tensor
    identity: CacheIdentity
    layer: int
    recent_k: Optional[torch.Tensor] = None
    recent_v: Optional[torch.Tensor] = None

    def detached(self):
        return LayerKV(self.k.detach(),self.v.detach(),self.identity,self.layer)


def validate(cache,identity,expected_tokens):
    for entry in cache:
        a,b=entry.identity,identity
        # Sample/step can change only when caller explicitly rebuilds/relabels;
        # even equal-shaped history from another trajectory/version is invalid.
        if a!=b or entry.k.shape[1]!=expected_tokens:
            raise ValueError('Stale/wrong-model/wrong-context functional KV cache')


def build_history(network,history,embeds,identity,recent_gradient=True,checkpoint=True,audit=False):
    if history.ndim!=5 or history.shape[0]!=1 or history.shape[1]%network.frames_per_chunk:
        raise ValueError('History must contain complete original latent chunks')
    cache=()
    for start in range(0,history.shape[1],network.frames_per_chunk):
        recent=start+network.frames_per_chunk==history.shape[1]
        grad=recent and recent_gradient and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad):
            result=network.forward_chunk(history[:,start:start+network.frames_per_chunk].detach(),embeds,
                identity.context_noise,cache,start,sparse=False,step=0,identity=identity,
                checkpoint=checkpoint,collect_aux=False)
            cache=result['cache']
        if not grad:cache=tuple(x.detached() for x in cache)
    validate(cache,identity,identity.history_end-identity.history_start)
    if audit:
        for entry in cache:
            for value in (entry.recent_k,entry.recent_v):
                if value is not None and value.requires_grad:value.retain_grad()
    return cache
