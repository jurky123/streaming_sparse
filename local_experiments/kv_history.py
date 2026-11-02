"""Equal-budget KV history policies using the unchanged native model forward."""
from contextlib import contextmanager

import torch

CHUNK_FRAMES = 3
HISTORY_FRAMES = 15
POLICIES = {
    'A_first_chunk': 'First complete chunk + latest 4 complete chunks',
    'B_latest_five': 'Latest 5 complete chunks',
    'C_first_three_starts': 'First latent frame of chunks 0, 1, 2 + latest 4 complete chunks',
}


def select_frames(policy, completed_chunks):
    """Sorted absolute latent-frame IDs; overlap is filled with recent history."""
    if policy not in POLICIES or completed_chunks < 0:
        raise ValueError((policy, completed_chunks))
    end = completed_chunks * CHUNK_FRAMES
    budget = min(HISTORY_FRAMES, end)
    if policy == 'B_latest_five':
        return list(range(end - budget, end))
    recent = set(range(max(0, end - 12), end))
    anchors = range(min(3, end)) if policy == 'A_first_chunk' else range(0, min(9, end), 3)
    selected = recent | set(anchors)
    for frame in range(end - 1, -1, -1):
        if len(selected) >= budget:
            break
        selected.add(frame)
    assert len(selected) == budget
    return sorted(selected)


def compact_cache(cache, previous_frames, completed_chunks, frame_tokens, policy):
    """Keep selected clean KV rows without recomputing or changing their RoPE."""
    current = list(range((completed_chunks - 1) * 3, completed_chunks * 3))
    available = previous_frames + current
    desired = select_frames(policy, completed_chunks)
    offsets = {frame: i for i, frame in enumerate(available)}
    if not set(desired) <= offsets.keys():
        raise AssertionError('Policy tried to resurrect an evicted frame')
    indices = torch.tensor([offsets[x] for x in desired], device=cache[0]['k'].device)
    valid = len(available) * frame_tokens
    retained = len(desired) * frame_tokens
    for layer in cache:
        assert layer['local_end_index'].item() == valid
        assert layer['global_end_index'].item() == completed_chunks * 3 * frame_tokens
        for kind in ('k', 'v'):
            source = layer[kind][:, :valid].unflatten(1, (len(available), frame_tokens))
            # index_select owns storage, so an overlapping in-place copy is safe.
            layer[kind][:, :retained].copy_(source.index_select(1, indices).flatten(1, 2))
        layer['local_end_index'].fill_(retained)
    return desired


@contextmanager
def audit_attention():
    """Observe the actual self-attention operands, including every model layer."""
    import wan.modules.causal_model as causal
    original = causal.attention
    state = dict(expected=None, calls=0, observed_tokens=[])

    def counted(q, k, v, *args, **kwargs):
        assert k.shape[1] == v.shape[1] == state['expected'], (k.shape, state)
        assert q.shape[1] <= k.shape[1]
        state['calls'] += 1
        state['observed_tokens'].append(k.shape[1])
        return original(q, k, v, *args, **kwargs)

    causal.attention = counted
    try:
        yield state
    finally:
        causal.attention = original


def forward(wrapper, current, embeds, timestep, cache, chunk, frame_tokens, audit):
    history = min(chunk * 3, HISTORY_FRAMES) * frame_tokens
    audit.update(expected=history + 3 * frame_tokens, calls=0, observed_tokens=[])
    t = torch.full((1, 3), float(timestep), device=current.device, dtype=torch.float32)
    result = wrapper(current, {'prompt_embeds': embeds}, t, kv_cache=cache,
                     crossattn_cache=[None] * len(cache), current_start=chunk * 3 * frame_tokens)[1]
    assert audit['calls'] == len(cache)
    assert all(layer['local_end_index'].item() == audit['expected'] for layer in cache)
    assert torch.isfinite(result).all(), 'Nonfinite generated latent'
    return result
