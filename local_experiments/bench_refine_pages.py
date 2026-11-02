"""Real QKV, fixed budget: legacy fused summaries + actual sparse kernel."""
import json
import math
import statistics
import torch
from migration.unified_router.opt19.prototype import segment
from .refine_pages import ROOT,make_model
from .learned_pool import Bank
from .learned_cache.data import records
from .learned_cache.model import output_for_masks
from .cache_sparse_prototype import load_kernel
from .response_hard import features,balanced
from .cache_grouping import reorder
from .learned_cache.io import atomic

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    choice=json.loads((ROOT/'calibration_base.json').read_text());base=make_model(choice['rep']).cuda()
    base.load_state_dict(torch.load(choice['path'],weights_only=False,map_location='cuda')['model'])
    cal=make_model('calibration').cuda();cal.load_state_dict(torch.load(ROOT/'calibration/best.pt',weights_only=False,map_location='cuda')['model'])
    # One fixed steady-state event; no claim of full-model/video throughput.
    path=next(p for p in records('test') if p.parent.parent.name=='skateboard' and p.name=='c20_l14_s2.pt')
    bank=Bank(4,max_history=1,max_events=1);e=bank.get(path);h=e['h'];data=torch.load(path,weights_only=True)
    q=data['q'].cuda();kh=h['k'].to(q.dtype);vh=h['v'].to(q.dtype);kc=data['k'].cuda();vc=data['v'].cuda()
    k=torch.cat((kh,kc),1).contiguous();v=torch.cat((vh,vc),1).contiguous();heads,n,d=q.shape;p=kh.shape[1]//64;tiles=n//64
    blhd=lambda x:x.transpose(0,1).unsqueeze(0).contiguous()
    qb,kb,vb=map(blhd,(q,k,v));kernel=load_kernel()
    w=base.weight[h['layer'],h['heads']].contiguous();bias=base.bias[h['layer'],h['heads']].contiguous()
    eye=torch.eye(128,device='cuda').expand(heads,-1,-1).contiguous()
    def build():
        left,lc,_=segment(kh,w,bias,eye[:,:,:64].contiguous(),False)
        right,lc2,_=segment(kh,w,bias,eye[:,:,64:].contiguous(),False)
        return torch.cat((left,right),-1).reshape(heads,p,4,d),lc.reshape(heads,p,4)
    pk,lc=build();count=lc.exp();fp=base(h)
    assert torch.allclose(pk.float(),fp[0],atol=.01,rtol=.003)
    assert torch.allclose(count,fp[2],atol=.001,rtol=.001)
    current=torch.arange(p,p+tiles,device='cuda').expand(heads,tiles,tiles)
    def route(calibrated=False,reference=False):
        routeq=q.float().reshape(heads,tiles,64,d).mean(2)
        keys=fp[0] if reference else pk.float();logs=fp[2].log() if reference else lc
        logits=(routeq@keys.flatten(1,2).transpose(1,2)/math.sqrt(d)).reshape(heads,tiles,p,4)+logs[:,None]
        z=logits.logsumexp(-1)
        if calibrated:z=z+cal.correction(routeq,h,(keys,None,logs.exp()))
        chosen=z.topk(72,-1).indices
        return torch.cat((chosen,current),-1).sort(-1).values.unsqueeze(0).contiguous()
    def sparse(calibrated=False):
        lut=route(calibrated);out=torch.empty_like(qb);lse=torch.empty((1,heads,n),device='cuda',dtype=torch.float32)
        kernel[(tiles,heads)](qb,kb,vb,d**-.5,lut.shape[-1],lut,lse,out,heads,n,k.shape[1],tiles,d,64,64,num_warps=4,num_stages=3)
        return out[0].transpose(0,1)
    def dense():return torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]
    denseout=dense();ids=torch.linspace(0,n-1,64,device='cuda').long();checks={}
    for label,calibrated in [('base',False),('calibrated',True)]:
        lut=route(calibrated)[0];out=sparse(calibrated).float();reference_lut=route(calibrated,True)[0]
        mask=torch.zeros_like(e['z'],dtype=torch.bool);mask[...,-1]=True
        selected=lut[:,ids//64,:72]
        # LUT is sorted; all history page IDs precede the current IDs.
        mask[...,:-1].scatter_(-1,selected,True)
        expected=output_for_masks(e['z'],e['values'],mask[:,:,None])[:,:,0]
        rel=(out[:,ids]-expected).square().sum()/expected.square().sum()
        assert rel<1e-4,rel
        mask_a=torch.zeros((heads,tiles,p),device='cuda',dtype=torch.bool).scatter_(-1,lut[:,:,:72],True)
        mask_b=torch.zeros_like(mask_a).scatter_(-1,reference_lut[:,:,:72],True)
        checks[label]=dict(kernel_vs_exact_selected_relative_mse=rel.item(),
            fp16_summary_mask_disagreement_fraction=(mask_a!=mask_b).float().mean().item(),
            sampled_nmse=((out[:,ids]-e['target']).square().mean(-1)/e['target'].square().mean(-1).clamp_min(1e-5)).mean().item())
    raw=torch.load(path.with_name(path.name[:-6]+'_history.pt'),weights_only=True)
    ck=raw['k'][:,-4608:].cuda().float();cv=raw['v'][:,-4608:].cuda().float()
    def group_new_chunk(calibrated=False):
        x=features(ck,cv,bank.dictionary[h['layer']],'response');order=balanced(x,[3,3,2,2,2])
        ordered_k=reorder(ck,order);ordered_v=reorder(cv,order)
        if calibrated:
            projected=(ordered_k@getattr(cal,f'basis_{h["layer"]}')).reshape(heads,-1,64,8)
            variance=(projected-projected.mean(2,keepdim=True)).square().mean(2)
            return ordered_k.to(q.dtype),ordered_v.to(q.dtype),variance
        return ordered_k.to(q.dtype),ordered_v.to(q.dtype)
    functions={'dense':dense,'sparse_base':lambda:sparse(False),'sparse_calibrated':lambda:sparse(True),
               'fused_summary_five_chunks':build,'group_new_chunk':group_new_chunk,
               'group_new_chunk_calibrated':lambda:group_new_chunk(True)}
    graphs={}
    for name,fn in functions.items():
        for _ in range(5):fn()
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):output=fn()
        graphs[name]=(graph,output)
    timings={name:[] for name in functions};names=list(functions)
    for i in range(30):
        for name in names[i%len(names):]+names[:i%len(names)]:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graphs[name][0].replay();b.record();b.synchronize();timings[name].append(a.elapsed_time(b))
    times={name:dict(median_ms=statistics.median(values),p10_ms=sorted(values)[3],p90_ms=sorted(values)[26]) for name,values in timings.items()}
    dense_ms=times['dense']['median_ms'];amortized=times['group_new_chunk']['median_ms']/4+times['fused_summary_five_chunks']['median_ms']/20
    calibrated_amortized=times['group_new_chunk_calibrated']['median_ms']/4+times['fused_summary_five_chunks']['median_ms']/20
    result=dict(event=str(path),heads=heads,query_tokens=n,history_tokens=kh.shape[1],selected_history_tokens=72*64,
        counts_fixed_by_age=True,checks=checks,cuda_graph_interleaved=times,
        estimated_steady_preprocess_ms_per_call=amortized,
        calibrated_estimated_steady_preprocess_ms_per_call=calibrated_amortized,
        module_speedup_cached={name:dense_ms/times[name]['median_ms'] for name in ['sparse_base','sparse_calibrated']},
        module_speedup_with_estimated_preprocessing={'sparse_base':dense_ms/(times['sparse_base']['median_ms']+amortized),
            'sparse_calibrated':dense_ms/(times['sparse_calibrated']['median_ms']+calibrated_amortized)},
        persistent_summary_k_bytes=pk.numel()*pk.element_size(),logcount_bytes=lc.numel()*lc.element_size(),
        calibration_variance_bytes=h['variance8'].numel()*h['variance8'].element_size(),
        notes=['Uses unchanged legacy opt19 prototype kernel twice to retain all128 dimensions, FP16 summary output.',
               'Actual LightForcing sparse forward kernel, full4608 query tokens,3 captured heads; not all12head/fullmodel/video benchmark.',
               'Prototype/V output correction swap is excluded from speed path; only initial mass routing and optional bounded calibration.',
               'Estimated steady preprocessing: one new clean-chunk grouping per4 denoise calls plus1/5 of measured five-chunk summary construction per4 calls.',
               'Preprocessing includes output BF16 conversion and, for calibrated variant, page-local8D variance construction; cache bookkeeping/eviction excluded.',
               'Cached-summary scenario stores extra K/count/variance auxiliary data; not yet equal total physical KV deployment.'])
    atomic(ROOT/'kernel_benchmark.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
