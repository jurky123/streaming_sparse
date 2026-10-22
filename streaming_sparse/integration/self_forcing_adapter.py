"""Scoped interception of Self-Forcing's cached self-attention call.

Original projections, RoPE, KV writes, rolling and output projection all execute
unchanged. Text cross-attention lives in another module and is never patched.
One adapter per process; inference batch size is explicitly restricted to one.
"""
import contextvars
import hashlib
import inspect
import math
import time
import torch

_active = contextvars.ContextVar('streaming_sparse_self_attention', default=None)


def positions_3d(flat, height, width):
    return torch.stack((flat // (height * width), (flat // width) % height, flat % width), -1)


class SelfForcingAdapter:
    def __init__(self, model, capture=None, router=None, keep_ratio=0.3,
                 layers=None, heads=None, context_noise=0):
        self.model, self.capture, self.router = model, capture, router
        self.keep_ratio, self.layers, self.heads = keep_ratio, layers, heads
        self.context_noise = context_noise
        self.handles, self.states, self.encoded = [], {}, {}
        self.metrics = []
        self.provenance = {}
        self.begin_prompt('unset')

    def begin_prompt(self, prompt_id):
        self.prompt_id = str(prompt_id)
        self.states.clear()
        self.encoded.clear()
        self.steps, self.chunks = {}, {}

    def generator_pre(self, module, args, kwargs):
        timestep = kwargs['timestep'].flatten()[0].item()
        start = int(kwargs.get('current_start', 0))
        phase = 'context' if timestep == self.context_noise else 'denoise'
        if start not in self.chunks:
            self.chunks[start] = len(self.chunks)
        key = (start, phase)
        step = self.steps.get(key, 0)
        self.steps[key] = step + 1
        self.event = dict(**self.provenance, prompt_id=self.prompt_id, chunk_id=self.chunks[start], step_id=step,
                          timestep=timestep, phase=phase)

    def _pre(self, layer, module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        cache = a['kv_cache']
        if cache is None:
            raise ValueError('Adapter requires actual streaming KV cache inference')
        if a['x'].shape[0] != 1:
            raise ValueError('v0.1 adapter supports batch size 1')
        _, height, width = a['grid_sizes'][0].tolist()
        n = a['x'].shape[1]
        start = int(a['current_start'])
        end = start + n
        old_global = int(cache['global_end_index'].item())
        old_local = int(cache['local_end_index'].item())
        size = cache['k'].shape[1]
        flat = self.states.get(layer)
        if flat is None:
            if old_local:
                raise ValueError('Attach adapter before starting inference to track cache positions')
            flat = torch.full((size,), -1, device=cache['k'].device, dtype=torch.long)
            self.states[layer] = flat
        evicted = 0
        if module.local_attn_size != -1 and end > old_global and n + old_local > size:
            evicted = n + old_local - size
            sink = module.sink_size * height * width
            rolled = old_local - evicted - sink
            if rolled < 0:
                raise ValueError('Cache too small for sink and current chunk')
            flat[sink:sink + rolled] = flat[sink + evicted:sink + evicted + rolled].clone()
        local_end = old_local + end - old_global - evicted
        local_start = local_end - n
        if local_start < 0 or local_end > size:
            raise ValueError('Invalid official KV cache bounds; reduce length or use local attention')
        flat[local_start:local_end] = torch.arange(start, end, device=flat.device)
        visible_start = max(0, local_end - module.max_attention_size)
        if visible_start > local_start:
            raise ValueError('Attention window must contain the entire current chunk')
        visible = flat[visible_start:local_end].clone()
        if (visible < 0).any():
            raise ValueError('Uninitialized visible cache positions')
        context = {**self.event, 'layer_id': layer,
                   'history_tokens': local_start - visible_start,
                   'positions': positions_3d(visible, height, width),
                   'query_positions': positions_3d(torch.arange(start, end, device=flat.device), height, width),
                   'history_signature': (start, tuple(visible[:local_start-visible_start].tolist()))}
        context['reset_token'] = _active.set(context)
        return None

    def _post(self, module, args, kwargs, output):
        context = _active.get()
        if context is not None:
            _active.reset(context['reset_token'])

    def __enter__(self):
        import wan.modules.causal_model as causal
        if getattr(causal.attention, '_streaming_sparse_patch', False):
            raise RuntimeError('Nested/concurrent adapters are unsupported')
        self.causal, self.original = causal, causal.attention
        def wrapped(q, k, v, *args, **kwargs):
            context = _active.get()
            if context is None:
                return self.original(q, k, v, *args, **kwargs)
            if args or kwargs:
                raise ValueError('Unexpected attention options; audit upstream implementation')
            if self.router is None or context['phase'] != 'denoise':
                output = self.original(q, k, v)
            else:
                output = self._sparse(context, q, k, v)
            if self.capture:
                if self.router is not None:
                    raise ValueError('Dense QKV capture and sparse generation must use separate runs')
                self.capture(context, q, k, v, output)
            return output
        wrapped._streaming_sparse_patch = True
        causal.attention = wrapped
        try:
            for layer, block in enumerate(self.model.blocks):
                pre = lambda m, a, kw, layer=layer: self._pre(layer, m, a, kw)
                self.handles.append(block.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
                self.handles.append(block.self_attn.register_forward_hook(self._post, with_kwargs=True, always_call=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exception):
        self.causal.attention = self.original
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def _sparse(self, context, q, k, v):
        nh, layer = context['history_tokens'], context['layer_id']
        if not nh or (self.layers is not None and layer not in self.layers):
            return self.original(q, k, v)
        output = torch.empty_like(q)
        for head in range(q.shape[2]):
            if self.heads is not None and head not in self.heads:
                output[:, :, head:head+1] = self.original(q[:, :, head:head+1], k[:, :, head:head+1], v[:, :, head:head+1])
                continue
            # Original history is immutable across denoising calls. Signature changes
            # on new chunk / window roll. Context passes stay dense and never cache H.
            key = (layer, head)
            signature = context['history_signature']
            tick = time.perf_counter()
            rebuilt = key not in self.encoded or self.encoded[key][0] != signature
            if rebuilt:
                state = self.router.precompute(k[0, :nh, head], v[0, :nh, head], context['positions'][:nh], head, layer)
                self.encoded[key] = (signature, state)
            state = self.encoded[key][1]
            pre_ms = (time.perf_counter() - tick) * 1000
            for start in range(0, q.shape[1], self.router.config.query_tile):
                tile = q[0, start:start+self.router.config.query_tile, head]
                route = self.router.route(tile, state, self.keep_ratio, relaxation=False)
                selected = (state['hard'] @ route['hard_mask']) > 0
                indices = torch.cat((torch.where(selected)[0], torch.arange(nh, len(k[0]), device=k.device)))
                # Real gather + softmax over retained raw KV; no statistics shortcut.
                result = self.original(tile[None, :, None], k[:, indices, head:head+1], v[:, indices, head:head+1])
                output[0, start:start+len(tile), head] = result[0, :, 0]
                self.metrics.append(dict(prompt_id=context['prompt_id'], layer=layer, head=head, chunk=context['chunk_id'], step=context['step_id'],
                    kept=int(selected.sum()), history=nh, precompute_host_ms=pre_ms if rebuilt else 0,
                    rebuilt=rebuilt))
                rebuilt = False
        return output
