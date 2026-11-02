"""Read-time temporal RoPE rebasing of the first chunk, with canonical KV storage."""
from contextlib import contextmanager
import math

import torch


def shift_temporal_(keys, tokens, freqs, offset):
    """Shift only the temporal complex pairs; spatial components remain bitwise equal."""
    if offset == 0 or tokens == 0:
        return keys
    complex_dim = keys.shape[-1] // 2
    temporal_pairs = complex_dim - 2 * (complex_dim // 3)
    part = keys[:, :tokens, :, :temporal_pairs * 2]
    z = torch.view_as_complex(part.double().reshape(*part.shape[:-1], temporal_pairs, 2))
    phase = freqs[offset, :temporal_pairs].to(keys.device)
    part.copy_(torch.view_as_real(z * phase).flatten(-2).to(keys.dtype))
    return keys


def sample_query_indices(height, width, device):
    y = torch.linspace(0, height - 1, min(4, height), device=device).round().long()
    x = torch.linspace(0, width - 1, min(4, width), device=device).round().long()
    spatial = (y[:, None] * width + x[None, :]).flatten()
    return torch.cat([spatial + t * height * width for t in range(3)])


def anchor_mass(q, keys, anchor_tokens, indices, alternative_anchor=None):
    """Exact FP32 softmax denominator over ALL keys for a fixed spatial query sample."""
    sampled = q[0].index_select(0, indices).transpose(0, 1).float()
    logits = torch.bmm(sampled, keys[0].permute(1, 2, 0).float()) / math.sqrt(q.shape[-1])
    anchor_lse = torch.logsumexp(logits[..., :anchor_tokens], dim=-1)
    rest_lse = torch.logsumexp(logits[..., anchor_tokens:], dim=-1)
    mass = (anchor_lse - torch.logaddexp(anchor_lse, rest_lse)).exp().mean(-1)
    alternative = None
    if alternative_anchor is not None:
        alternative_logits = torch.bmm(sampled, alternative_anchor[0].permute(1, 2, 0).float()) / math.sqrt(q.shape[-1])
        alternative_lse = torch.logsumexp(alternative_logits, dim=-1)
        alternative = (alternative_lse - torch.logaddexp(alternative_lse, rest_lse)).exp().mean(-1)
    return mass, alternative


class AnchorMonitor:
    def __init__(self, wrapper, audit, variant, grid_height, grid_width):
        if variant not in ('baseline', 'rebased'):
            raise ValueError(variant)
        self.wrapper, self.audit, self.variant = wrapper, audit, variant
        self.frame_tokens = grid_height * grid_width
        self.indices = sample_query_indices(grid_height, grid_width, next(wrapper.parameters()).device)
        self.rows = []

    def begin(self, chunk, step, kind):
        self.chunk, self.step, self.kind = chunk, step, kind
        self.offset = max(0, chunk * 3 - 15)

    @contextmanager
    def install(self):
        import wan.modules.causal_model as causal
        original = causal.attention

        def observed(q, k, v, *args, **kwargs):
            layer = self.audit['calls']
            # Both variants use an identically sized temporary attention operand.
            # The persistent cache is NEVER rotated or duplicated.
            working = k.clone()
            tokens = 3 * self.frame_tokens if self.chunk else 0
            if self.variant == 'rebased':
                shift_temporal_(working, tokens, self.wrapper.model.freqs, self.offset)
            if tokens:
                alternative = None
                if self.variant == 'baseline':
                    alternative = k[:, :tokens].clone()
                    shift_temporal_(alternative, tokens, self.wrapper.model.freqs, self.offset)
                mass, counterfactual = anchor_mass(q, working, tokens, self.indices, alternative)
                self.rows.append(dict(chunk=self.chunk, step=self.step, kind=self.kind, layer=layer,
                                      kv_tokens=k.shape[1], anchor_tokens=tokens,
                                      virtual_anchor_start=self.offset if self.variant == 'rebased' else 0,
                                      query_count=len(self.indices), mass=mass,
                                      same_state_rebased_mass=counterfactual))
            return original(q, working, v, *args, **kwargs)

        causal.attention = observed
        try:
            yield self
        finally:
            causal.attention = original

    def flush(self):
        rows = self.rows
        for row in rows:
            row['mass'] = row['mass'].cpu().tolist()
            if row['same_state_rebased_mass'] is not None:
                row['same_state_rebased_mass'] = row['same_state_rebased_mass'].cpu().tolist()
        self.rows = []
        return rows
