"""Pinned public assets. Direct network; no experiment execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Also get T5/tokenizer and VAE for recollection and video decoding')
    parser.add_argument('--endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    for key in list(os.environ):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            os.environ.pop(key, None)
    os.environ.pop('HF_HUB_OFFLINE', None)
    from huggingface_hub import hf_hub_download
    revisions = json.loads((ROOT / 'migration/model_revisions.json').read_text())
    wan = json.loads((ROOT / 'migration/wan_assets.reference.json').read_text())
    backbone = json.loads((ROOT / 'migration/backbone_audit.json').read_text())
    selected = []
    for row in wan['files']:
        name = row['path']
        if name in ('config.json', 'diffusion_pytorch_model.safetensors') or (
            args.full and (name in ('models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth') or name.startswith('google/umt5-xxl/'))
        ):
            selected.append(('Wan-AI/Wan2.1-T2V-1.3B', name, ROOT / 'wan_models/Wan2.1-T2V-1.3B', row['sha256']))
    selected.append(('gdhe17/Self-Forcing', 'checkpoints/self_forcing_dmd.pt', ROOT, backbone['sha256']))
    for repo, name, local, expected in selected:
        target = local / name
        if not target.is_file() or digest(target) != expected:
            print(f'Download {repo}@{revisions[repo]} / {name}', flush=True)
            hf_hub_download(repo, name, revision=revisions[repo], local_dir=local, endpoint=args.endpoint)
        if digest(target) != expected:
            raise ValueError(f'SHA256 mismatch: {target}')
        print(f'OK {target.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
