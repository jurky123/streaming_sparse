"""Read-only peer sampling during measurements; never stops other GPU tasks."""
import threading,time
from .gap_group_isolated import device_peers
class PeerWatch:
    def __init__(self,interval=2.):
        self.interval=interval;self.stop=threading.Event();self.samples=[];self.started=time.monotonic()
    def sample(self):
        try:self.samples.append(dict(seconds=time.monotonic()-self.started,peers=device_peers()))
        except Exception as exc:self.samples.append(dict(seconds=time.monotonic()-self.started,error=repr(exc)))
    def __enter__(self):
        self.sample()
        def run():
            while not self.stop.wait(self.interval):self.sample()
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample()
    def summary(self):return dict(process_scope='compute and graphics via NVIDIA XML',interval_seconds=self.interval,samples=self.samples,scope='No observed peers at sampled times; short unobserved overlap is still possible.')
    def validate(self):
        bad=[x for x in self.samples if x.get('peers') or x.get('error')]
        if bad:raise RuntimeError(f'GPU peer sampling rejected measurement: {bad}')
