from pathlib import Path
from ..v03.paths import protect_output as legacy_protect


def protect_output(path):
    out=legacy_protect(path)
    old=Path(__file__).resolve().parents[1]/'v03'
    if out.is_relative_to(old):
        raise ValueError('v0.4 outputs must not modify the frozen v0.3 tree')
    return out
