"""Paired native-RoPE vs dynamic-anchor-RoPE rollout with shared saved random inputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch

from .anchor_rebase import AnchorMonitor
from .kv_history import audit_attention, forward, compact_cache, select_frames
from .run_kv_history import ROOT, sha, write_json, status, decode_videos, inspect_video


@torch.no_grad()
def generate(wrapper, inputs, steps, context_noise, variant, folder, height, width):
    from streaming_sparse.v08.backbone import native_cache
    from streaming_sparse.capture.capture_runner import sdpa_backend
    noise, transitions, embeds = (inputs[name] for name in ('initial_noise', 'transition_noise', 'prompt_embeds'))
    chunks = noise.shape[1] // 3
    frame_tokens = (height // 16) * (width // 16)
    capacity = 18 * frame_tokens
    assert all(b.self_attn.local_attn_size == -1 and b.self_attn.max_attention_size >= capacity
               for b in wrapper.model.blocks)
    cache = native_cache(wrapper, 'cuda', torch.bfloat16, tokens=capacity)
    physical_bytes = sum(x[k].numel() * x[k].element_size() for x in cache for k in ('k', 'v'))
    retained, outputs, trace = [], [], []
    torch.cuda.reset_peak_memory_stats()
    with (folder / f'{variant}.attention.jsonl').open('w') as metrics:
        with sdpa_backend(), audit_attention() as audit:
            monitor = AnchorMonitor(wrapper, audit, variant, height // 16, width // 16)
            with monitor.install():
                for chunk in range(chunks):
                    assert retained == select_frames('A_first_chunk', chunk)
                    offset = max(0, chunk * 3 - 15) if variant == 'rebased' else 0
                    row = dict(chunk=chunk, history_latent_frame_ids=list(retained),
                               history_tokens_per_layer=len(retained) * frame_tokens,
                               current_tokens_per_layer=3 * frame_tokens,
                               physical_kv_buffer_bytes_all_layers=physical_bytes,
                               virtual_anchor_frames=[offset + i for i in range(3)] if chunk else [],
                               query_frame_positions=[3 * chunk + i for i in range(3)], forwards=[])
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    current = noise[:, chunk * 3:(chunk + 1) * 3]
                    for step, timestep in enumerate(steps):
                        monitor.begin(chunk, step, 'denoise')
                        x0 = forward(wrapper, current, embeds, timestep, cache, chunk, frame_tokens, audit)
                        row['forwards'].append(dict(kind='denoise', timestep=timestep,
                                                    self_attention_calls=audit['calls'],
                                                    actual_kv_tokens_per_layer=audit['expected']))
                        if step < len(steps) - 1:
                            current = wrapper.scheduler.add_noise(
                                x0.flatten(0, 1), transitions[chunk, step].flatten(0, 1),
                                torch.full((3,), steps[step + 1], device='cuda')).unflatten(0, (1, 3))
                    monitor.begin(chunk, len(steps), 'clean_context')
                    forward(wrapper, x0, embeds, context_noise, cache, chunk, frame_tokens, audit)
                    row['forwards'].append(dict(kind='clean_context', timestep=context_noise,
                                                self_attention_calls=audit['calls'],
                                                actual_kv_tokens_per_layer=audit['expected']))
                    retained = compact_cache(cache, retained, chunk + 1, frame_tokens, 'A_first_chunk')
                    row['retained_after_write'] = list(retained)
                    row['retained_tokens_per_layer_after_write'] = len(retained) * frame_tokens
                    outputs.append(x0.cpu())
                    for item in monitor.flush():
                        metrics.write(json.dumps(item) + '\n')
                    metrics.flush()
                    torch.cuda.synchronize()
                    row['seconds_including_telemetry'] = time.perf_counter() - tick
                    trace.append(row)
                    write_json(folder / f'{variant}.kv_trace.json', trace)
                    status(folder, 'generating', variant=variant, chunk=chunk + 1, chunks=chunks,
                           virtual_anchor_frames=row['virtual_anchor_frames'],
                           seconds=round(row['seconds_including_telemetry'], 3))
    latent = torch.cat(outputs, 1)
    torch.save(latent, folder / f'{variant}.latents.pt')
    result = dict(generation_seconds_including_telemetry=sum(x['seconds_including_telemetry'] for x in trace),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  physical_persistent_kv_buffer_bytes_all_layers=physical_bytes,
                  steady_history_kv_bytes_all_layers=physical_bytes * 15 // 18)
    del cache
    torch.cuda.empty_cache()
    return latent, trace, result


def side_by_side(folder, frames, fps, width, height):
    from contextlib import ExitStack
    import av
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    with ExitStack() as stack:
        readers = [stack.enter_context(av.open(str(folder / f'{name}.mp4'))) for name in ('baseline', 'rebased')]
        decoders = [r.decode(video=0) for r in readers]
        writer = stack.enter_context(imageio.get_writer(
            folder / 'comparison.mp4', fps=fps, codec='libx264', pixelformat='yuv420p',
            macro_block_size=None, ffmpeg_log_level='error', output_params=['-crf', '18', '-threads', '4']))
        for index in range(frames):
            canvas = Image.new('RGB', (width * 2, height + 48), (20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            for panel, (decoder, title) in enumerate(zip(decoders, ['Before: original anchor time', 'After: anchor time rebased to window start'])):
                canvas.paste(Image.fromarray(next(decoder).to_ndarray(format='rgb24')), (panel * width, 48))
                draw.text((panel * width + 8, 12), title, font=font, fill='white')
            writer.append_data(np.asarray(canvas))
        assert all(next(d, None) is None for d in decoders)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior', default='artifacts/kv_history_30s_20261029_seed0')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    os.chdir(ROOT)
    prior = Path(a.prior).resolve()
    parent = json.loads((prior / 'manifest.json').read_text())
    assert sha(prior / 'shared_inputs.pt') == parent['shared_inputs_sha256']
    assert sha(ROOT / 'checkpoints/self_forcing_dmd.pt') == parent['checkpoint_sha256']
    for name, digest in parent['source_sha256'].items():
        assert sha(Path(__file__).parent / name) == digest, name
    preflight = ROOT / 'local_setup/results/anchor_rebase_preflight.json'
    assert json.loads(preflight.read_text())['passed']
    folder = Path(a.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(preflight, folder / 'preflight.json')
    shutil.copyfile(prior / 'prompt.txt', folder / 'prompt.txt')
    manifest = dict(parent_experiment=str(prior), parent_manifest_sha256=sha(prior / 'manifest.json'),
                    seed=parent['seed'], prompt=parent['prompt'], height=parent['height'], width=parent['width'],
                    seconds=parent['seconds'], fps=parent['fps'], frames=parent['output_frames'], chunks=parent['chunks'],
                    checkpoint_sha256=parent['checkpoint_sha256'], shared_inputs_sha256=parent['shared_inputs_sha256'],
                    policy='First complete chunk + latest 4 complete chunks, 15 historical latent frames.',
                    intervention='Only anchor K temporal RoPE is shifted by max(0, current_chunk_start - 15). Canonical KV cache remains at original positions. Spatial K channels, Q and V are unchanged by the intervention.',
                    allocation='Both variants clone one attention K operand per layer call. No additional persistent KV is stored; anchor-key counterfactual diagnostics use a transient slice.',
                    measurement='Exact FP32 softmax mass over all retained/current keys for 4x4 uniformly spaced spatial queries per each of 3 query frames (48 queries), all heads/layers, all steps. Plots average heads/layers/4 denoise steps; clean-context measurements are saved separately.',
                    same_state_counterfactual='On the baseline trajectory, rebase only anchor K and recompute its attention mass with the same Q, remaining K and full softmax denominator; this does not affect generation.',
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    source_sha256={x.name: sha(x) for x in Path(__file__).parent.glob('*anchor_rebase*.py')})
    write_json(folder / 'manifest.json', manifest)
    torch.set_num_threads(4)
    torch.manual_seed(parent['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status(folder, 'loading_generator')
    from streaming_sparse.v08.backbone import load_teacher, native_steps
    wrapper = load_teacher()
    wrapper.model.num_frame_per_block = 3
    steps, config = native_steps(wrapper)
    assert steps == parent['denoising_steps'] and config['context_noise'] == parent['context_noise']
    inputs = {k: v.cuda() for k, v in torch.load(prior / 'shared_inputs.pt', map_location='cpu', weights_only=True).items()}
    latents, traces, results = {}, {}, {}
    for variant in ('baseline', 'rebased'):
        latents[variant], traces[variant], results[variant] = generate(
            wrapper, inputs, steps, config['context_noise'], variant, folder, parent['height'], parent['width'])
        if variant == 'baseline':
            previous = torch.load(prior / 'A_first_chunk.latents.pt', map_location='cpu', weights_only=True)
            assert torch.equal(latents[variant], previous), 'Instrumented baseline differs from prior experiment'
            del previous
    assert torch.equal(latents['baseline'][:, :18], latents['rebased'][:, :18])
    assert not torch.equal(latents['baseline'][:, 18:], latents['rebased'][:, 18:])
    for before, after in zip(traces['baseline'], traces['rebased']):
        for field in ('history_latent_frame_ids', 'history_tokens_per_layer', 'current_tokens_per_layer',
                      'physical_kv_buffer_bytes_all_layers', 'retained_after_write',
                      'retained_tokens_per_layer_after_write', 'forwards'):
            assert before[field] == after[field], field
    write_json(folder / 'audit.json', dict(passed=True, exact_match_to_previous_baseline=True,
               identical_first_latent_frames=18, intervention_starts_chunk_zero_based=6,
               intervention_starts_video_second=69 / 16, equal_kv_budget_every_forward=True,
               equal_retained_frame_ids=True, persistent_kv_buffers_equal=True,
               attention_calls_verified=2 * parent['chunks'] * 5 * len(wrapper.model.blocks)))
    del wrapper, inputs
    gc.collect()
    torch.cuda.empty_cache()
    shutil.copyfile(prior / 'A_first_chunk.mp4', folder / 'baseline.mp4')
    decode_videos({'rebased': latents['rebased']}, folder, parent['output_frames'], parent['fps'])
    status(folder, 'building_comparison')
    side_by_side(folder, parent['output_frames'], parent['fps'], parent['width'], parent['height'])
    videos = [inspect_video(folder / f'{name}.mp4') for name in ('baseline', 'rebased', 'comparison')]
    assert all(v['frames'] == parent['output_frames'] and v['fps'] == parent['fps'] and
               abs(v['duration'] - parent['seconds']) < 0.001 for v in videos)
    write_json(folder / 'result.json', dict(completed=True, videos=videos, variants=results))
    status(folder, 'completed', output=str(folder))


if __name__ == '__main__':
    main()
