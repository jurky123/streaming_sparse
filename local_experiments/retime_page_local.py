"""Interleaved timing to avoid comparing serial warmup/load transients."""
import statistics
import torch
from .audit_page_local import ROOT
from .page_local import make_model,prototypes
from .learned_pool import Bank,actions
from .learned_cache.data import records
from .learned_cache.model import predict,initial_mask
from .learned_cache.io import atomic

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    e=Bank(4,max_history=1,max_events=1).get(records('test')[0]);functions={}
    for name,rep in [('mixture_hard','mixture'),('mixture_reconstruction','mixture'),('pool_hard','pool')]:
        model=make_model(rep).cuda();model.load_state_dict(torch.load(ROOT/name/'best.pt',weights_only=False,map_location='cuda')['model'])
        functions[name+'_preprocess']=lambda m=model,r=rep:prototypes(m,e['h'],r)
        proto=prototypes(model,e['h'],rep)
        for mode in ['none','residual']:
            def route(p=proto,k=mode):
                z,v,_=predict(e['route'],p,e['kc'],e['vc'])
                return initial_mask(z) if k=='none' else actions(z,v,k)
            functions[name+'_'+mode]=route
    for _ in range(10):
        for fn in functions.values():fn()
    results={k:[] for k in functions};names=list(functions)
    for i in range(30):
        for k in names[i%len(names):]+names[:i%len(names)]:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();functions[k]();b.record();b.synchronize();results[k].append(a.elapsed_time(b))
    result={k:dict(median_ms=statistics.median(v),p10_ms=sorted(v)[3],p90_ms=sorted(v)[26]) for k,v in results.items()}
    atomic(ROOT/'interleaved_timing.json',result);print(result)

if __name__=='__main__':main()
