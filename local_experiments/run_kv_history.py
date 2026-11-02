"""Generate paired 30-second videos with three equal-budget native KV policies."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from .kv_history import POLICIES, HISTORY_FRAMES, select_frames, compact_cache, audit_attention, forward

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def status(folder, phase, **kwargs):
    value = dict(phase=phase, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    write_json(folder / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


@torch.no_grad()
def generate(wrapper, embeds, noise, transitions, steps, context_noise, policy, folder):
    from streaming_sparse.v08.backbone import native_cache
    from streaming_sparse.capture.capture_runner import sdpa_backend
    chunks = noise.shape[1] // 3
    frame_tokens = (noise.shape[-2] // 2) * (noise.shape[-1] // 2)
    capacity = (HISTORY_FRAMES + 3) * frame_tokens
    assert all(b.self_attn.local_attn_size == -1 and b.self_attn.max_attention_size >= capacity
               for b in wrapper.model.blocks)
    cache = native_cache(wrapper, 'cuda', torch.bfloat16, tokens=capacity)
    physical_bytes = sum(x[k].numel() * x[k].element_size() for x in cache for k in ('k', 'v'))
    bytes_per_token = physical_bytes // capacity
    retained, outputs, trace = [], [], []
    torch.cuda.reset_peak_memory_stats()
    with sdpa_backend(), audit_attention() as audit:
        for chunk in range(chunks):
            assert retained == select_frames(policy, chunk)
            history_tokens = len(retained) * frame_tokens
            row = dict(chunk=chunk, history_latent_frame_ids=list(retained),
                       history_tokens_per_layer=history_tokens, current_tokens_per_layer=3 * frame_tokens,
                       history_kv_bytes_all_layers=history_tokens * bytes_per_token,
                       physical_kv_buffer_bytes_all_layers=physical_bytes, forwards=[])
            torch.cuda.synchronize()
            tick = time.perf_counter()
            current = noise[:, chunk * 3:(chunk + 1) * 3]
            for step, timestep in enumerate(steps):
                x0 = forward(wrapper, current, embeds, timestep, cache, chunk, frame_tokens, audit)
                row['forwards'].append(dict(kind='denoise', timestep=timestep,
                                            self_attention_calls=audit['calls'],
                                            actual_kv_tokens_per_layer=audit['expected']))
                if step + 1 < len(steps):
                    current = wrapper.scheduler.add_noise(
                        x0.flatten(0, 1), transitions[chunk, step].flatten(0, 1),
                        torch.full((3,), steps[step + 1], device='cuda')).unflatten(0, (1, 3))
            # Overwrite current scratch KV with the clean-context forward before retention.
            forward(wrapper, x0, embeds, context_noise, cache, chunk, frame_tokens, audit)
            row['forwards'].append(dict(kind='clean_context', timestep=context_noise,
                                        self_attention_calls=audit['calls'],
                                        actual_kv_tokens_per_layer=audit['expected']))
            retained = compact_cache(cache, retained, chunk + 1, frame_tokens, policy)
            row['retained_after_write'] = list(retained)
            row['retained_tokens_per_layer_after_write'] = len(retained) * frame_tokens
            outputs.append(x0.cpu())
            torch.cuda.synchronize()
            row['seconds'] = time.perf_counter() - tick
            trace.append(row)
            write_json(folder / f'{policy}.kv_trace.json', trace)
            status(folder, 'generating', policy=policy, chunk=chunk + 1, chunks=chunks,
                   history_tokens_per_layer=history_tokens, chunk_seconds=round(row['seconds'], 3))
    latent = torch.cat(outputs, 1)
    torch.save(latent, folder / f'{policy}.latents.pt')
    result = dict(generation_seconds=sum(x['seconds'] for x in trace),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  physical_kv_buffer_bytes_all_layers=physical_bytes,
                  steady_history_kv_bytes_all_layers=HISTORY_FRAMES * frame_tokens * bytes_per_token)
    del cache
    torch.cuda.empty_cache()
    return latent, trace, result


@torch.no_grad()
def decode_videos(latents, folder, frames, fps):
    import imageio.v2 as imageio
    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    for policy, latent in latents.items():
        vae.model.clear_cache()
        position = 0
        with imageio.get_writer(folder / f'{policy}.mp4', fps=fps, codec='libx264',
                               pixelformat='yuv420p', macro_block_size=None,
                               ffmpeg_log_level='error', output_params=['-crf', '18', '-threads', '4']) as writer:
            for start in range(0, latent.shape[1], 3):
                rgb = vae.decode_to_pixel(latent[:, start:start + 3].cuda(), use_cache=True)[0]
                assert torch.isfinite(rgb).all()
                pixels = ((rgb[:frames - position].permute(0, 2, 3, 1) + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()
                for frame in pixels:
                    writer.append_data(frame)
                position += len(pixels)
                if start % 15 == 0 or position == frames:
                    status(folder, 'decoding', policy=policy, frames_written=position, frames=frames)
                if position >= frames:
                    break
        assert position == frames
    del vae
    gc.collect()
    torch.cuda.empty_cache()


def inspect_video(path):
    import av
    with av.open(str(path)) as reader:
        stream = reader.streams.video[0]
        result = dict(path=path.name, width=stream.width, height=stream.height,
                      fps=float(stream.average_rate), duration=float(stream.duration * stream.time_base))
        result['frames'] = sum(1 for _ in reader.decode(video=0))
    return result


def compare_video(folder, frames, fps, width, height):
    import av
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from contextlib import ExitStack
    titles = ['A: First chunk + latest 4', 'B: Latest 5 chunks', 'C: First frames of chunks 1-3 + latest 4']
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    with ExitStack() as stack:
        containers = [stack.enter_context(av.open(str(folder / f'{policy}.mp4'))) for policy in POLICIES]
        decoders = [c.decode(video=0) for c in containers]
        writer = stack.enter_context(imageio.get_writer(
            folder / 'comparison.mp4', fps=fps, codec='libx264', pixelformat='yuv420p',
            macro_block_size=None, ffmpeg_log_level='error', output_params=['-crf', '18', '-threads', '4']))
        for index in range(frames):
            canvas = Image.new('RGB', (width * 3, height + 48), (20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            for panel, (decoder, title) in enumerate(zip(decoders, titles)):
                canvas.paste(Image.fromarray(next(decoder).to_ndarray(format='rgb24')), (panel * width, 48))
                draw.text((panel * width + 8, 12), title, font=font, fill='white')
            writer.append_data(np.asarray(canvas))
        for decoder in decoders:
            assert next(decoder, None) is None


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--prompt-file', default='local_setup/prompt.txt')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--fps', type=int, default=16)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    a = p.parse_args()
    os.chdir(ROOT)
    assert a.seconds > 0 and a.height % 16 == 0 and a.width % 16 == 0
    preflight = ROOT / 'local_setup/results/kv_history_preflight.json'
    assert json.loads(preflight.read_text())['passed']
    folder = Path(a.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(preflight, folder / 'preflight.json')
    prompt = Path(a.prompt_file).read_text().strip()
    assert prompt and '\n' not in prompt
    (folder / 'prompt.txt').write_text(prompt + '\n')
    frames = a.seconds * a.fps
    chunks = math.ceil((frames + 3) / 12)
    frame_tokens = (a.height // 16) * (a.width // 16)
    manifest = dict(prompt=prompt, seed=a.seed, height=a.height, width=a.width, fps=a.fps,
                    seconds=a.seconds, output_frames=frames, generated_frames=chunks * 12 - 3,
                    chunks=chunks, latent_frames_per_chunk=3, history_budget_latent_frames=15,
                    history_tokens_per_layer=15 * frame_tokens,
                    current_tokens_per_layer=3 * frame_tokens, policies=POLICIES,
                    frame_semantics='First frame means the first latent frame, including all of its spatial KV tokens.',
                    warmup='Deduplicate overlapping anchors/recent chunks, then fill with most recent unselected history until min(15, available frames).',
                    positions='Original absolute temporal RoPE preserved; chronological KV row order.',
                    model='Self-Forcing 1.3B generator_ema; native dense SDPA on retained KV, all 30 layers/12 heads.',
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    checkpoint_sha256=sha(ROOT / 'checkpoints/self_forcing_dmd.pt'),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256={x.name: sha(x) for x in Path(__file__).parent.glob('*kv_history*.py')})
    write_json(folder / 'manifest.json', manifest)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status(folder, 'loading_text_encoder')
    from utils.wan_wrapper import WanTextEncoder
    encoder = WanTextEncoder().cuda().to(torch.bfloat16).eval().requires_grad_(False)
    embeds = encoder([prompt])['prompt_embeds']
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    from streaming_sparse.v08.backbone import load_teacher, native_steps
    status(folder, 'loading_generator')
    wrapper = load_teacher()
    wrapper.model.num_frame_per_block = 3
    steps, config = native_steps(wrapper)
    manifest['denoising_steps'] = steps
    manifest['context_noise'] = config['context_noise']
    shape = (16, a.height // 8, a.width // 8)
    rng = torch.Generator(device='cuda').manual_seed(a.seed)
    noise = torch.randn((1, chunks * 3, *shape), device='cuda', dtype=torch.bfloat16, generator=rng)
    transitions = torch.stack([torch.stack([
        torch.randn((1, 3, *shape), device='cuda', dtype=torch.bfloat16, generator=rng)
        for _ in range(len(steps) - 1)]) for _ in range(chunks)])
    torch.save(dict(initial_noise=noise.cpu(), transition_noise=transitions.cpu(),
                    prompt_embeds=embeds.cpu()), folder / 'shared_inputs.pt')
    manifest['shared_inputs_sha256'] = sha(folder / 'shared_inputs.pt')
    write_json(folder / 'manifest.json', manifest)
    latents, traces, results = {}, {}, {}
    for policy in POLICIES:
        latents[policy], traces[policy], results[policy] = generate(
            wrapper, embeds, noise, transitions, steps, config['context_noise'], policy, folder)
    reference = latents['A_first_chunk']
    for policy, latent in latents.items():
        # The first six chunks see the same complete history, before any policy-dependent eviction.
        assert torch.equal(reference[:, :18], latent[:, :18]), (policy, 'prefix mismatch')
        for left, right in zip(traces['A_first_chunk'], traces[policy]):
            for field in ('history_tokens_per_layer', 'current_tokens_per_layer',
                          'history_kv_bytes_all_layers', 'physical_kv_buffer_bytes_all_layers',
                          'retained_tokens_per_layer_after_write', 'forwards'):
                assert left[field] == right[field], (policy, field)
    write_json(folder / 'kv_budget_audit.json', dict(
        passed=True, exact_equal_budget_every_forward=True, exact_equal_physical_buffers=True,
        layers=len(wrapper.model.blocks), forwards_per_chunk=len(steps) + 1,
        total_verified_attention_calls=3 * chunks * (len(steps) + 1) * len(wrapper.model.blocks),
        identical_first_latent_frames=18, identical_first_decoded_frames=69,
        first_policy_dependent_chunk_zero_based=6, history_tokens_per_layer=15 * frame_tokens,
        current_tokens_per_layer=3 * frame_tokens, policies=list(POLICIES)))
    del wrapper, embeds, noise, transitions
    gc.collect()
    torch.cuda.empty_cache()
    decode_videos(latents, folder, frames, a.fps)
    status(folder, 'building_comparison')
    compare_video(folder, frames, a.fps, a.width, a.height)
    videos = [inspect_video(folder / f'{policy}.mp4') for policy in POLICIES]
    videos.append(inspect_video(folder / 'comparison.mp4'))
    assert all(v['frames'] == frames and v['fps'] == a.fps and abs(v['duration'] - a.seconds) < 0.001 for v in videos)
    write_json(folder / 'result.json', dict(completed=True, videos=videos, variants=results,
                                           equal_kv_budget_verified=True, same_noise_and_prompt=True))
    status(folder, 'completed', output=str(folder))


if __name__ == '__main__':
    main()
