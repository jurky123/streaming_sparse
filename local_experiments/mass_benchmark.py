"""FP32 eager route-only timing: include query pooling, scores and top72."""
import json,math,statistics
import torch
from .mass_common import ROOT,Engine,cases
from .mass_router import Events,make_model,scores
from .learned_cache.io import atomic,digest,RunLock
LAYERS=[0,4,14,17,24,29]

def measure(fn):
    for _ in range(15):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(7):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(20):fn()
        end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
    return dict(median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples))

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    folder=ROOT/'train_00_30'
    assert json.loads((folder/'status.json').read_text())['status']=='completed'
    saved=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
    dest=ROOT/'bench';lock=RunLock(dest)
    identity=dict(code=digest(__file__),weights=digest(folder/'best.pt'),gpu=torch.cuda.get_device_name(),layers=LAYERS)
    if (dest/'identity.json').exists():assert json.loads((dest/'identity.json').read_text())==identity
    else:atomic(dest/'identity.json',identity)
    engine=Engine();events=Events(LAYERS)
    def callback(current,q,k,v):
        layer=current['layer'];path=dest/f'layer{layer:02d}.json'
        if current['step']!=0 or layer not in LAYERS or path.exists():return
        e=events.event(current,q,k,v);model=make_model(layer,'mass_var').cuda().eval()
        model.load_state_dict(saved['models']['mass_var'][layer]);proto=model(e['h'])
        pk,_,count=proto;flat=pk.flatten(1,2);squared=flat.square();log_count=count.log()[:,None]
        full=q[0].transpose(0,1).float().reshape(12,72,64,128)
        def mean_route():return scores(proto,full.mean(2)).topk(72,-1).indices
        def var_route():return scores(proto,full.mean(2),full.var(2,unbiased=False),model.alpha).topk(72,-1).indices
        def cached_route():
            mean=full.mean(2);variance=full.var(2,unbiased=False)
            logits=(mean@flat.transpose(1,2)/math.sqrt(128)).reshape(12,72,360,4)+log_count
            corr=(variance@squared.transpose(1,2)/256).reshape_as(logits)
            return (logits+model.alpha[:,None,None,None]*corr).logsumexp(-1).topk(72,-1).indices
        assert torch.equal(var_route(),cached_route())
        h=e['h'];incoming={key:h[key][:,-4608:] for key in ['k','v']}
        result=dict(layer=layer,mean=measure(mean_route),variance_dynamic_k2=measure(var_route),variance_cached_k2=measure(cached_route),
            incoming_chunk_aggregation_only=measure(lambda:model(incoming)),cached_k2_bytes=squared.numel()*squared.element_size())
        atomic(path,result);print(json.dumps(result),flush=True)
    try:
        engine.replay(cases('val')[0],callback,stop_chunk=5)
        rows=[json.loads((dest/f'layer{l:02d}.json').read_text()) for l in LAYERS]
        metrics={m:statistics.median(r[m]['median_ms'] for r in rows) for m in ['mean','variance_dynamic_k2','variance_cached_k2','incoming_chunk_aggregation_only']}
        atomic(ROOT/'routing_benchmark.json',dict(identity=identity,rows=rows,median_across_layers_ms=metrics,
            dynamic_ratio=metrics['variance_dynamic_k2']/metrics['mean'],cached_ratio=metrics['variance_cached_k2']/metrics['mean'],
            scope='FP32 eager PyTorch; batch12 heads, all72 query tiles, 360 history pages,4proto; same learned variance model weights for all score ablations. Mean/variance pooling and top72 included. Cached variant also reuses log-count. No grouping, no sparse attention kernel, no whole-model timing. Shared machine; not isolated hardware.'))
    finally:engine.callback=None;lock.close()
if __name__=='__main__':run()
