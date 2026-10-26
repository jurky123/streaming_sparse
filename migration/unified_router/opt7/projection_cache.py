"""Cache projected history prototypes while the exact history Part objects live.

The current chunk is always reprojected. Strong references and object identity
prevent allocator pointer reuse from making a different history appear equal.
"""
from ..core import Sketch
class HistoryProjection:
    def __init__(self,prepare):self.prepare=prepare;self.entries={}
    def __call__(self,sketch,kind):
        key=(kind,sketch.key_basis.data_ptr());history=sketch.parts[:-1]
        cached=self.entries.get(key)
        def subset(parts):return Sketch(parts,sketch.costs,sketch.origins,sketch.history_tokens,sketch.current_tokens,sketch.key_basis,sketch.value_basis)
        if cached is None or len(cached[0])!=len(history)or any(a is not b for a,b in zip(cached[0],history)):
            projected=self.prepare(subset(history),kind)[1]
            self.entries[key]=(history,projected)
        else:projected=cached[1]
        basis,current=self.prepare(subset(sketch.parts[-1:]),kind)
        return basis,projected+current
