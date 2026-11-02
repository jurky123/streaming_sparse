"""Reproducible 30 s, same-weight comparison under a 15-latent-frame KV cap."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from .baseline_methods import BaselineAdapter, METHODS, EXTERNAL
from .run_kv_history import ROOT, sha, write_json, status, decode_videos, inspect_video

PROMPTS = {
    'sailboat': (ROOT / 'local_setup/prompt.txt').read_text().strip(),
    'skateboard': 'A young woman wearing a bright yellow jacket and blue jeans rides a skateboard along a sunlit seaside promenade. The camera tracks smoothly beside her as she pushes off, turns gently, and passes white railings. Realistic cinematic video, consistent clothing and face, natural continuous motion.',
    'dog': 'A golden retriever wearing a red bandana runs across a green park, chasing a blue ball. The dog catches up to the ball, nudges it, and continues running while the camera follows at ground level. Warm afternoon daylight, realistic fur, continuous lively motion, trees in the background.',
}


@torch.no_grad()
def prepare(folder, cases):
    from utils.wan_wrapper import WanTextEncoder
    missing = [case for case in cases if not (folder / case / 'shared_inputs.pt').exists()]
    if not missing:
        return
    prior = ROOT / 'artifacts/kv_history_30s_20261029_seed0/shared_inputs.pt'
    embeds = {'sailboat': torch.load(prior, map_location='cpu', weights_only=True)['prompt_embeds']}
    other = [case for case in missing if case != 'sailboat']
    if other:
        status(folder, 'loading_text_encoder')
        encoder = WanTextEncoder().cuda().to(torch.bfloat16).eval().requires_grad_(False)
        for case in other:
            embeds[case] = encoder([PROMPTS[case]])['prompt_embeds'].cpu()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    for case in missing:
        target = folder / case
        target.mkdir(exist_ok=True)
        generator = torch.Generator(device='cuda').manual_seed(0)
        shape = (16, 64, 96)
        noise = torch.randn((1, 123, *shape), device='cuda', dtype=torch.bfloat16, generator=generator)
        transitions = torch.stack([torch.stack([
            torch.randn((1, 3, *shape), device='cuda', dtype=torch.bfloat16, generator=generator)
            for _ in range(3)]) for _ in range(41)])
        torch.save(dict(initial_noise=noise.cpu(), transition_noise=transitions.cpu(),
                        prompt_embeds=embeds[case]), target / 'shared_inputs.pt')
        (target / 'prompt.txt').write_text(PROMPTS[case] + '\n')
        write_json(target / 'inputs.json', dict(seed=0, prompt=PROMPTS[case],
                   sha256=sha(target / 'shared_inputs.pt'), latent_frames=123, output_frames=480))
        del noise, transitions
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def rollout(wrapper, inputs, method, folder, steps, context_noise, chunks):
    from streaming_sparse.capture.capture_runner import sdpa_backend
    adapter = BaselineAdapter(wrapper, method, 1536, chunks=41)
    cap_bytes = 15 * 1536 * 12 * 128 * 2 * 2 * 30
    outputs, trace = [], []
    torch.cuda.reset_peak_memory_stats()
    with sdpa_backend(), adapter.install():
        for chunk in range(chunks):
            torch.cuda.synchronize()
            start = time.perf_counter()
            current = inputs['initial_noise'][:, chunk * 3:chunk * 3 + 3]
            before = adapter.storage()
            for step, ts in enumerate(steps):
                adapter.begin(chunk, step)
                t = torch.full((1, 3), float(ts), device='cuda', dtype=torch.float32)
                x0 = wrapper(current, {'prompt_embeds': inputs['prompt_embeds']}, t,
                       kv_cache=[{}] * 30, crossattn_cache=[None] * 30, current_start=chunk * 3 * 1536)[1]
                assert adapter.calls == 30 and torch.isfinite(x0).all()
                if step < 3:
                    current = wrapper.scheduler.add_noise(x0.flatten(0, 1),
                        inputs['transition_noise'][chunk, step].flatten(0, 1),
                        torch.full((3,), steps[step + 1], device='cuda')).unflatten(0, (1, 3))
            adapter.begin(chunk, 4, clean=True)
            wrapper(x0, {'prompt_embeds': inputs['prompt_embeds']},
                    torch.full((1, 3), float(context_noise), device='cuda', dtype=torch.float32),
                    kv_cache=[{}] * 30, crossattn_cache=[None] * 30, current_start=chunk * 3 * 1536)
            assert adapter.calls == 30
            adapter.end_chunk()
            after = adapter.storage()
            assert after['persistent_kv_bytes'] <= cap_bytes, (method, chunk, after)
            outputs.append(x0.cpu())
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            trace.append(dict(chunk=chunk, seconds=seconds, before=before, after=after))
            write_json(folder / f'{method}.timing.json', trace)
            status(folder, 'generating', method=method, chunk=chunk+1, chunks=chunks,
                   seconds=round(seconds, 3), kv_gib=round(after['persistent_kv_bytes']/2**30, 3))
    latent = torch.cat(outputs, 1)
    torch.save(latent, folder / f'{method}.latents.pt')
    write_json(folder / f'{method}.attention_trace.json', adapter.rows)
    result = dict(method=method, generated_chunks=chunks,
                  generation_seconds_including_first_shape_compilation=sum(r['seconds'] for r in trace),
                  steady_chunk_seconds=[r['seconds'] for r in trace[6:]],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  max_persistent_kv_bytes=max(r['after']['persistent_kv_bytes'] for r in trace),
                  max_auxiliary_state_bytes=max(r['after']['auxiliary_state_bytes'] for r in trace),
                  kv_cap_bytes=cap_bytes, attention_layer_calls=chunks*5*30,
                  denoise_attention_pairs=sum(r['attention_pairs'] for r in adapter.rows if not r['clean']),
                  denoise_full_cap_pairs=sum(r['full_cap_pairs'] for r in adapter.rows if not r['clean']),
                  equal_upper_bound_verified=True)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return latent, result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='artifacts/ar_sparse_baselines_20261029')
    p.add_argument('--cases', nargs='+', choices=PROMPTS, default=list(PROMPTS))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--chunks', type=int, default=41)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    os.chdir(ROOT)
    folder = Path(a.output).resolve()
    folder.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert json.loads((ROOT/'local_setup/results/baseline_methods_preflight.json').read_text())['passed']
    prepare(folder, a.cases)
    if a.prepare_only:
        return
    versions = {name: subprocess.check_output(['git', '-C', str(EXTERNAL/name), 'rev-parse', 'HEAD'], text=True).strip()
                for name in ['LightForcing', 'Forcing-KV', 'DASA']}
    manifest = dict(date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seed=0,
        height=512, width=768, fps=16, seconds=30, chunks=a.chunks,
        historical_kv_cap_latent_frames=15, historical_kv_cap_tokens_per_head_per_layer=23040,
        current_chunk_tokens=4608, historical_kv_cap_bytes_all_layers=4246732800,
        checkpoint_sha256=sha(ROOT/'checkpoints/self_forcing_dmd.pt'),
        checkpoint='Self-Forcing 1.3B generator_ema', official_commits=versions,
        cases=a.cases, methods=a.methods, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__), ROOT/'local_experiments/baseline_methods.py']},
        limitations=[
          'Shared GPU with other workloads: timings are observed wall times, not isolated speed claims.',
          'Light Forcing CAG/HSA transfer to identical unfine-tuned weights; NOT the paper tuned checkpoint result.',
          'Light Forcing official Triton backend on H200, FP32-sum syntax compatibility patch; no dense fallback.',
          'Light Forcing uses common 3-frame anchor and 18-frame attention extent; target sparsity .8/base .9, keep_frames 6/keep_sink 3/keep_near 2, block size 64.',
          'DASA-KV only: official memory selection, thresholds .85/.3 and max age 4, added FIFO 15-frame cap; no region early exit or training adaptation.',
          'Forcing-KV default per-group sink/local/dynamic budget 1/1/1 frames; smaller ACTUAL budget, common upper bound only. Native candidate design limits dynamic context to 1..3 frames.',
          'Forcing-KV common native RoPE/SDPA and torch head-gather, official policy/function bodies and published head classification; dead warmup full-head copies released.',
          'Three prompts and one seed are a screening comparison, not full VBench or statistical evidence.'
        ])
    write_json(folder/'manifest.json', manifest)
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'],
                   stdout=(folder/'gpu_at_start.csv').open('w'), check=True)
    from streaming_sparse.v08.backbone import load_teacher, native_steps
    status(folder, 'loading_generator')
    wrapper = load_teacher()
    wrapper.model.num_frame_per_block = 3
    steps, config = native_steps(wrapper)
    for case in a.cases:
        target = folder / case
        info = json.loads((target/'inputs.json').read_text())
        assert info['sha256'] == sha(target/'shared_inputs.pt')
        inputs = {k:v.cuda() for k,v in torch.load(target/'shared_inputs.pt', map_location='cpu', weights_only=True).items()}
        for method in a.methods:
            done = target/f'{method}.result.json'
            if done.exists() and json.loads(done.read_text()).get('completed'):
                continue
            latent, result = rollout(wrapper, inputs, method, target, steps, config['context_noise'], a.chunks)
            frames = min(480, a.chunks*12-3)
            tick = time.perf_counter()
            decode_videos({method:latent}, target, frames, 16)
            result['decode_seconds'] = time.perf_counter()-tick
            result['video'] = inspect_video(target/f'{method}.mp4')
            assert result['video']['frames'] == frames
            result['completed'] = True
            write_json(done, result)
            del latent
            gc.collect()
            torch.cuda.empty_cache()
        del inputs
    status(folder, 'completed', cases=a.cases, methods=a.methods)


if __name__ == '__main__':
    main()
