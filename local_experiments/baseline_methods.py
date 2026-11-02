"""Same-backbone adapters. Official sources are pinned under external_baselines.

LightForcing calls the unmodified official Triton router and sparse kernel.
Forcing-KV calls official attention/cache function bodies, with common native
RoPE and SDPA and torch head gathering to isolate cache-policy differences.
DASA-KV calls the official retention policy; region reuse is explicitly excluded.
"""
import ast
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import types

import torch
import torch.nn.functional as F

from .anchor_rebase import shift_temporal_
from .kv_history import select_frames

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'external_baselines'
METHODS = ('fifo', 'anchor', 'rebase', 'lightforcing', 'dasa_kv', 'forcingkv')


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_definitions(path, names, namespace):
    """Load unchanged selected AST bodies, avoiding unrelated framework imports."""
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.ClassDef):
            selected.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names)
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def dense(q, k, v, *args, **kwargs):
    if q.shape[2] == 0:
        return torch.empty_like(q)
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), dropout_p=0., is_causal=False).transpose(1, 2)


def tensor_bytes(items):
    """Physical storage, deduplicated across aliases/views."""
    seen = {}
    for x in items:
        if isinstance(x, torch.Tensor) and x.numel():
            storage = x.untyped_storage()
            seen[(str(x.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())


class BaselineAdapter:
    def __init__(self, wrapper, method, frame_tokens, chunks=41):
        from wan.modules.causal_model import causal_rope_apply
        self.wrapper, self.method, self.ft = wrapper, method, frame_tokens
        self.rope = causal_rope_apply
        self.budget = 15 * frame_tokens
        self.layers = len(wrapper.model.blocks)
        self.cache = [dict(k=None, v=None, frames=[], reference=None, age=None, frame_number=0)
                      for _ in wrapper.model.blocks]
        self.rows = []
        if method == 'lightforcing':
            assert frame_tokens % 64 == 0, 'Official hierarchical router needs frame-aligned blocks'
            self.kernel = load_file('official_lightforcing_kernel', EXTERNAL / 'LightForcing/wan/modules/kernel.py')
            self.sparsities = official_definitions(
                EXTERNAL / 'LightForcing/wan/modules/sparse_attention.py',
                ['calculate_chunk_sparsities'], {'math': math})['calculate_chunk_sparsities'](
                    chunks * 3, 3, 18, {'sparsity': .8, 'sparsity_base': .9})
        if method == 'dasa_kv':
            self.dasa = load_file('official_dasa_kv', EXTERNAL / 'DASA/DASA/dasa/kv_compression.py')
        if method == 'forcingkv':
            self._init_forcing()

    def _init_forcing(self):
        from wan.modules.model import WanRMSNorm
        def gather(q, k, v, temporal, spatial):
            return tuple(z[:, :, ids].contiguous() for ids in (spatial, temporal) for z in (q, k, v))
        def batched_rope(x, grid, freqs, start_frame):
            # Official fused kernel broadcasts one grid across concatenated Q/K.
            # Native rope instead iterates over the batch dimension of grid.
            if grid.shape[0] != x.shape[0]:
                assert x.shape[0] % grid.shape[0] == 0
                grid = grid.repeat(x.shape[0] // grid.shape[0], 1)
            return self.rope(x, grid, freqs, start_frame=start_frame)
        names = ['_allocate_fixed_cache', '_allocate_ring_cache', '_ring_append_',
                 '_materialize_ring_view', '_fill_cache_from_chunk_indices_',
                 'CausalWanSelfAttention', '_load_offline_head_groups', '_apply_offline_head_allocation']
        import os
        import torch.distributed as dist
        self.fns = official_definitions(EXTERNAL / 'Forcing-KV/wan/modules/causal_model_forcingkv.py', names,
                    dict(torch=torch, nn=torch.nn, F=F, math=math, json=json, os=os, dist=dist,
                         WanRMSNorm=WanRMSNorm, attention=self.forcing_attention,
                         rope_apply_triton=batched_rope, extract_heads_triton=gather))
        args = types.SimpleNamespace(ar_start=1, spatial_context_length=1,
                    temporal_context_length=1, dynamic_context_length=1,
                    dynamic_temporal_enabled=True, num_frame_patch=6,
                    sim_retention_ratio=.33, quantization_enabled=False,
                    offline_head_file=str(EXTERNAL / 'Forcing-KV/configs_head/forcingkv_heads_selfforcing.json'))
        self.forcing_modules = []
        self.forcing_cache = []
        for block in self.wrapper.model.blocks:
            original = block.self_attn
            # Share weights, never allocate or train a second model.
            att = self.fns['CausalWanSelfAttention'].__new__(self.fns['CausalWanSelfAttention'])
            torch.nn.Module.__init__(att)
            for key in ('q', 'k', 'v', 'o', 'norm_q', 'norm_k', 'num_heads', 'head_dim', 'dim'):
                setattr(att, key, getattr(original, key))
            att.sink_size, att.args, att.patch_dynamic_score = 1, args, None
            self.forcing_modules.append(att)
            empty = next(original.parameters()).new_empty((1, 0, att.num_heads, att.head_dim))
            self.forcing_cache.append(dict(sink_k=empty, sink_v=empty, local_k=empty, local_v=empty))
        self.proxy = types.SimpleNamespace(args=args, _offline_head_groups=None,
                    num_heads=self.forcing_modules[0].num_heads, num_layers=self.layers)
        self.proxy._load_offline_head_groups = types.MethodType(self.fns['_load_offline_head_groups'], self.proxy)

    def forcing_attention(self, q, k, v):
        self.call_pairs += q.shape[1] * k.shape[1] * q.shape[2]
        self.call_heads += q.shape[2]
        assert k.shape[1] - q.shape[1] <= self.budget
        return dense(q, k, v)

    def begin(self, chunk, step, clean=False):
        self.chunk, self.step, self.clean = chunk, step, clean
        self.calls = 0

    def end_chunk(self):
        if self.method == 'forcingkv' and self.chunk == 0:
            self.fns['_apply_offline_head_allocation'](self.proxy, self.forcing_cache)
            # The official allocator leaves unused full-head warmup tensors alive.
            # Release only these dead copies; active group buffers are unchanged.
            for c in self.forcing_cache[1:]:
                for key in ('sink_k', 'sink_v', 'local_k', 'local_v'):
                    c.pop(key, None)

    def storage(self):
        if self.method == 'forcingkv':
            kv = [v for c in self.forcing_cache for key, v in c.items()
                  if key.endswith(('_k', '_v'))]
            aux = [getattr(a, 'patch_dynamic_score', None) for a in self.forcing_modules]
        else:
            kv = [c[key] for c in self.cache for key in ('k', 'v')]
            aux = [c[key] for c in self.cache for key in ('reference', 'age')]
        return dict(persistent_kv_bytes=tensor_bytes(kv), auxiliary_state_bytes=tensor_bytes(aux))

    def forward(self, layer, att, x, seq_lens, grid_sizes, freqs, block_mask,
                kv_cache=None, current_start=0, cache_start=None):
        self.calls += 1
        if self.method == 'forcingkv':
            self.call_pairs, self.call_heads = 0, 0
            result = self.forcing_modules[layer](x, seq_lens, grid_sizes, freqs, block_mask,
                    kv_cache=self.forcing_cache[layer], current_start=current_start,
                    layer_idx=layer, update_kv_cache=self.clean)
            assert self.call_heads == att.num_heads
            self.rows.append(dict(chunk=self.chunk, step=self.step, clean=self.clean, layer=layer,
                             attention_pairs=self.call_pairs, full_cap_pairs=x.shape[1] * (self.budget + x.shape[1]) * att.num_heads))
            return result
        b, s = x.shape[:2]
        shape = (b, s, att.num_heads, att.head_dim)
        q, k = att.norm_q(att.q(x)).view(shape), att.norm_k(att.k(x)).view(shape)
        v = att.v(x).view(shape)
        q = self.rope(q, grid_sizes, freqs, start_frame=self.chunk * 3).type_as(v)
        k = self.rope(k, grid_sizes, freqs, start_frame=self.chunk * 3).type_as(v)
        c = self.cache[layer]
        past_k = k[:, :0] if c['k'] is None else c['k']
        past_v = v[:, :0] if c['v'] is None else c['v']
        history_tokens = past_k.shape[1]
        assert history_tokens <= self.budget
        all_k, all_v = torch.cat((past_k, k), 1), torch.cat((past_v, v), 1)
        if self.method == 'dasa_kv' and self.clean:
            features = x.reshape(1, 3, self.ft, -1)
            if c['age'] is None:
                c['age'] = torch.zeros((1, self.ft), dtype=torch.long, device=x.device)
            if c['reference'] is not None:
                features = torch.cat((c['reference'], features), 1)
                candidate_k = torch.cat((past_k[:, -self.ft:], k), 1)
                candidate_v = torch.cat((past_v[:, -self.ft:], v), 1)
                old_k, old_v = past_k[:, :-self.ft], past_v[:, :-self.ft]
            else:
                candidate_k, candidate_v = k, v
                old_k, old_v = past_k, past_v
            threshold = self.dasa.layer_similarity_threshold(.85, .3, layer, self.layers)
            mask, c['age'], c['frame_number'] = self.dasa.compute_retention_mask(
                features, tail_frames=1, threshold=threshold, last_update_step=c['age'],
                current_frame_num=c['frame_number'], max_token_age=4, training=False)
            all_k = torch.cat((old_k, candidate_k[:, mask.flatten()]), 1)
            all_v = torch.cat((old_v, candidate_v[:, mask.flatten()]), 1)
            # FIFO cap is an explicit long-rollout adaptation of official DASA-KV.
            all_k, all_v = all_k[:, -self.budget:], all_v[:, -self.budget:]
            c['k'], c['v'] = all_k.clone(), all_v.clone()
            c['reference'] = features[:, -1:].clone()
        working_k = all_k
        if self.method == 'rebase' and self.chunk:
            working_k = all_k.clone()
            shift_temporal_(working_k, 3 * self.ft, freqs, max(0, self.chunk * 3 - 15))
        pairs = q.shape[1] * all_k.shape[1] * att.num_heads
        if self.method == 'lightforcing' and self.chunk:
            ratio = 1 - self.sparsities[self.chunk]
            smap, lut, topk = self.kernel.get_sm_80_120_block_map(
                q.contiguous(), all_k.contiguous(), ratio, 64, 64,
                frame_seq=self.ft, keep_frames=6, keep_sink=3, keep_near=2)
            assert 0 < topk < math.ceil(all_k.shape[1] / 64)
            result = self.kernel._attention.apply(q.contiguous(), all_k.contiguous(), all_v.contiguous(),
                                                  smap, lut, topk, 64, 64, None)
            pairs = q.shape[1] * topk * 64 * att.num_heads
        else:
            result = dense(q, working_k, all_v)
        self.rows.append(dict(chunk=self.chunk, step=self.step, clean=self.clean, layer=layer,
                         history_tokens=history_tokens, attended_tokens=all_k.shape[1],
                         attention_pairs=pairs, full_cap_pairs=q.shape[1] * (self.budget + q.shape[1]) * att.num_heads))
        if self.clean and self.method != 'dasa_kv':
            policy = 'B_latest_five' if self.method == 'fifo' else 'A_first_chunk'
            available = c['frames'] + list(range(self.chunk * 3, self.chunk * 3 + 3))
            desired = select_frames(policy, self.chunk + 1)
            indices = torch.tensor([available.index(f) for f in desired], device=x.device)
            c['k'] = all_k.unflatten(1, (len(available), self.ft)).index_select(1, indices).flatten(1, 2)
            c['v'] = all_v.unflatten(1, (len(available), self.ft)).index_select(1, indices).flatten(1, 2)
            c['frames'] = desired
        return att.o(result.flatten(2))

    @contextmanager
    def install(self):
        originals = []
        for layer, block in enumerate(self.wrapper.model.blocks):
            att = block.self_attn
            originals.append(att.forward)
            def wrapped(module, *args, _layer=layer, **kwargs):
                return self.forward(_layer, module, *args, **kwargs)
            att.forward = types.MethodType(wrapped, att)
        try:
            yield self
        finally:
            for block, original in zip(self.wrapper.model.blocks, originals):
                block.self_attn.forward = original
