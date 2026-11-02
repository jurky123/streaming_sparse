"""Download the original pinned assets using this machine's configured network."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    revisions = json.loads((ROOT / 'migration/model_revisions.json').read_text())
    reference = json.loads((ROOT / 'migration/wan_assets.reference.json').read_text())
    backbone = json.loads((ROOT / 'migration/backbone_audit.json').read_text())
    selected = []
    for row in reference['files']:
        name = row['path']
        if name in ('config.json', 'diffusion_pytorch_model.safetensors',
                    'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth') or name.startswith('google/umt5-xxl/'):
            selected.append(('Wan-AI/Wan2.1-T2V-1.3B', name,
                             ROOT / 'wan_models/Wan2.1-T2V-1.3B', row['sha256']))
    selected.append(('gdhe17/Self-Forcing', 'checkpoints/self_forcing_dmd.pt', ROOT, backbone['sha256']))
    verified = []
    for repo, name, directory, expected in selected:
        target = directory / name
        if not target.is_file() or digest(target) != expected:
            if target.is_symlink():
                raise ValueError(f'Refusing to overwrite a mismatched shared model: {target}')
            print(f'Download {repo}@{revisions[repo]} / {name}', flush=True)
            hf_hub_download(repo, name, revision=revisions[repo], local_dir=str(directory),
                            endpoint=args.endpoint, token=False, force_download=target.exists())
        if digest(target) != expected:
            raise ValueError(f'SHA256 mismatch: {target}')
        verified.append(dict(path=str(target.relative_to(ROOT)), sha256=expected,
                             bytes=target.stat().st_size, reused_symlink=target.is_symlink()))
        print(f'OK {target.relative_to(ROOT)}', flush=True)
    output = ROOT / 'local_setup/results'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models.json').write_text(json.dumps(verified, indent=2) + '\n')


if __name__ == '__main__':
    main()
