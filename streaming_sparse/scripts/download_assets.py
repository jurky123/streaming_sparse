"""Download pinned public model assets; uses process proxy environment."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
root = Path(__file__).resolve().parents[2]
records = {}
for repo, local, files in [
    ('Wan-AI/Wan2.1-T2V-1.3B', root / 'wan_models/Wan2.1-T2V-1.3B', None),
    ('gdhe17/Self-Forcing', root, ['checkpoints/self_forcing_dmd.pt']),
]:
    info = HfApi().model_info(repo)
    records[repo] = info.sha
    print(f'Downloading {repo} at {info.sha}', flush=True)
    if files is None:
        snapshot_download(repo, revision=info.sha, local_dir=local, max_workers=4)
    else:
        for name in files:
            hf_hub_download(repo, name, revision=info.sha, local_dir=local)
(root / 'streaming_sparse/artifacts/model_revisions.json').write_text(json.dumps(records, indent=2))
