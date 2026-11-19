from pathlib import Path
from fcntl import flock, LOCK_EX
from vbench2_beta_long.utils import split_video_into_clips

from paths import RUN
root=RUN/'segments'

def ensure_segments(source,name):
    cache=root/name
    cache.mkdir(exist_ok=True)
    ready=cache/'.ready'
    if ready.exists():
        return cache
    with (cache/'.split.lock').open('a') as lock:
        flock(lock,LOCK_EX)
        if not ready.exists():
            split_video_into_clips(str(source),str(root),duration=2)
            assert len(list(cache.glob('*.mp4')))==15
            ready.touch()
    return cache
