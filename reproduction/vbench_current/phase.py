import torch
from local_experiments.anchor_rebase import shift_temporal_

def set_phase(rt,canonical,offset):
    # Shift original frame-0 K from scratch, in global coordinates. With an
    # 18-frame window and 3-frame chunk the Q-to-anchor distances are 15..17.
    k=canonical.clone()
    shift_temporal_(k,1536,rt.engine.wrapper.model.freqs,offset)
    for l,cache in enumerate(rt.engine.cache):
        cache['k'][0,:1536].copy_(k[l])
        if l in rt.stores:
            def grouped(x):
                return x.transpose(0,1).reshape(12,24,64,128)
            summary=rt.summary_graph(rt.model[l],grouped(k[l]),grouped(cache['v'][0,:1536]))
            for field in ('keys','stats','detail','mean'):
                getattr(rt.stores[l].summary,field)[:,:24].copy_(getattr(summary,field))
            if l in rt.pset_layers:
                rt.write_encoding(l,rt.encode(l,summary),start=0)
