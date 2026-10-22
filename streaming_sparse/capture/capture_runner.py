"""Run the unchanged official pipeline: dense, QKV capture, or sparse evaluation."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import time
import torch


@contextmanager
def sdpa_backend():
    """Length-aware fallback for official cross-attention when flash-attn is absent."""
    import wan.modules.model as model
    import wan.modules.attention as attention_module
    original = model.flash_attention
    flags = attention_module.FLASH_ATTN_2_AVAILABLE, attention_module.FLASH_ATTN_3_AVAILABLE
    def sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
             q_scale=None, causal=False, window_size=(-1, -1), **kwargs):
        if q_lens is not None or window_size != (-1, -1):
            raise ValueError('SDPA fallback supports unpadded queries and full key windows')
        if q_scale is not None:
            q = q * q_scale
        mask = None
        if k_lens is not None:
            if causal:
                raise ValueError('Combined causal and key-length mask requires explicit implementation')
            lengths = torch.as_tensor(k_lens, device=k.device)
            mask = torch.arange(k.shape[1], device=k.device)[None, None, None] < lengths[:, None, None, None]
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale).transpose(1, 2).contiguous()
    model.flash_attention = sdpa
    attention_module.FLASH_ATTN_2_AVAILABLE = False
    attention_module.FLASH_ATTN_3_AVAILABLE = False
    try:
        yield
    finally:
        model.flash_attention = original
        attention_module.FLASH_ATTN_2_AVAILABLE, attention_module.FLASH_ATTN_3_AVAILABLE = flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['dense','capture','sparse'], default='capture')
    p.add_argument('--prompts', default='prompts/MovieGenVideoBench_extended.txt')
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--frames', type=int, default=12, help='Latent frames; multiple of 3, max 21 for global cache')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seeds', type=int, nargs='+', help='Cartesian product: use these exact generation seeds for every prompt')
    p.add_argument('--layers', type=int, nargs='+', default=[4,14,24])
    p.add_argument('--heads', type=int, nargs='+', default=[0,3,6,9])
    p.add_argument('--query-tile', type=int, default=16)
    p.add_argument('--tiles-per-event', type=int, default=4, help='0 captures all Q tiles')
    p.add_argument('--output', required=True)
    p.add_argument('--router-checkpoint')
    p.add_argument('--keep', type=float, default=0.3)
    p.add_argument('--config', default='configs/self_forcing_dmd.yaml')
    p.add_argument('--checkpoint', default='checkpoints/self_forcing_dmd.pt')
    p.add_argument('--backend', choices=['auto','sdpa'], default='auto')
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--local-attn-frames', type=int, default=-1)
    p.add_argument('--sink-frames', type=int, default=0)
    args = p.parse_args()
    if args.frames % 3 or args.frames < 3 or (args.frames > 21 and args.local_attn_frames == -1):
        p.error('Use a positive multiple of 3, <=21 for the default global cache')
    if args.mode == 'sparse' and not args.router_checkpoint:
        p.error('--router-checkpoint is required for sparse inference')
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
    from .hook_attention import AttentionCapture
    from ..integration.self_forcing_adapter import SelfForcingAdapter
    from ..router.model import load_router
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.merge(OmegaConf.load('configs/default_config.yaml'), OmegaConf.load(args.config))
    config.model_kwargs.local_attn_size = args.local_attn_frames
    config.model_kwargs.sink_size = args.sink_frames
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    pipeline = CausalInferencePipeline(config, device=torch.device('cuda'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    pipeline.generator.load_state_dict(checkpoint['generator_ema'], strict=True)
    del checkpoint
    pipeline.eval().requires_grad_(False).to(dtype=torch.bfloat16, device='cuda')
    capture = AttentionCapture(output, args.layers, args.heads, args.query_tile, args.tiles_per_event) if args.mode == 'capture' else None
    router = load_router(args.router_checkpoint, 'cuda').eval() if args.mode == 'sparse' else None
    adapter = SelfForcingAdapter(pipeline.generator.model, capture=capture, router=router,
        keep_ratio=args.keep, layers=args.layers, heads=args.heads, context_noise=config.context_noise)
    backend = sdpa_backend() if args.backend == 'sdpa' or not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE) else nullcontext()
    prompts = Path(args.prompts).read_text().splitlines()
    selected = prompts[args.offset:args.offset+args.limit]
    if not selected or any(not s.strip() for s in selected):
        raise ValueError('Select nonempty prompts')
    run = dict(arguments=vars(args), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
               denoising_steps=pipeline.denoising_step_list.tolist(), prompt_ids=[], runs=[])
    run_id = str(time.time_ns())
    metadata = output / 'metadata'
    metadata.mkdir(exist_ok=True)
    jobs = [(i, prompt, sample_seed)
            for i, prompt in enumerate(selected, args.offset)
            for sample_seed in (args.seeds if args.seeds is not None else [args.seed+i])]
    with torch.inference_mode(), backend, adapter:
        handle = pipeline.generator.register_forward_pre_hook(adapter.generator_pre, with_kwargs=True)
        try:
            for i, prompt, sample_seed in jobs:
                prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:20]
                adapter.begin_prompt(prompt_id)
                adapter.provenance = dict(seed=sample_seed, run_id=run_id)
                torch.manual_seed(sample_seed)
                noise = torch.randn((1, args.frames, 16, 60, 104), device='cuda', dtype=torch.bfloat16)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                video, latents = pipeline.inference(noise, [prompt], return_latents=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                name = f'{prompt_id}.seed{sample_seed}' if args.seeds is not None else prompt_id
                torch.save(latents.cpu(), output / f'{name}.latents.pt')
                if args.save_video:
                    import imageio.v3 as iio
                    pixels = (video[0].permute(0,2,3,1).float().cpu().clamp(0,1)*255).to(torch.uint8).numpy()
                    iio.imwrite(output / f'{name}.mp4', pixels, fps=16)
                pipeline.vae.model.clear_cache()
                run['prompt_ids'].append(prompt_id)
                run['runs'].append(dict(prompt_id=prompt_id, seed=sample_seed, seconds=elapsed))
                print(f'{args.mode}: prompt {i}, seed {sample_seed}, {elapsed:.2f}s', flush=True)
                (output / 'run.json').write_text(json.dumps(run, indent=2))
                (metadata / f'{run_id}.json').write_text(json.dumps(run, indent=2))
        finally:
            handle.remove()
    if capture:
        run['dense_backend_max_relative_error'] = max(capture.errors, default=None)
    if router:
        with (output / 'streaming_metrics.jsonl').open('w') as f:
            for row in adapter.metrics:
                f.write(json.dumps(row) + '\n')
    (output / 'run.json').write_text(json.dumps(run, indent=2))
    (metadata / f'{run_id}.json').write_text(json.dumps(run, indent=2))


if __name__ == '__main__':
    main()
