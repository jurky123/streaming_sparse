"""Source and input identities for paired checkpoint/evaluation provenance."""
import hashlib
from pathlib import Path
from ..v03.evaluate import source_identity as original_identity


def source_identity():
    h=hashlib.sha256(original_identity().encode())
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    h.update((Path(__file__).parent/'configs/protocol.json').read_bytes())
    return h.hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
