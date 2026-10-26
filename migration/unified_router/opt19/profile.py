"""Current combined path, innermost-scope kernel attribution and paired timing."""
import argparse,collections,gzip,json,statistics
import torch
from ..common import config,atomic_json
from ..opt18.setup import ROOT as P
from ..opt17.data import Events
from ..opt17.model import JointGroupedRouter
from ..opt18.measure import Runner,Opt14
from ..opt16.grouping import order,gather,restore
from ..opt5.probe import capture,timing
from .pipeline import Pipeline
from .route_fast import scores
from .tile import select
from .validate import native_mask
from .attention import attention,native
from .probe import ROOT,measure,nmse

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','shared'],default='shared');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    _,m=config(P);ev=Events(m,24);prompt=sorted({x['row']['prompt_id'] for x in ev.val})[0]
    x=next(x for x in ev.val if x['row']['prompt_id']==prompt and x['row']['history_tokens']==14040 and x['row']['step']==3 and x['row']['generation_seed']==0)
    model=JointGroupedRouter(m,24,'Fast35').cuda()
    model.load_state_dict(torch.load(P/'training/layer24/Fast35/best.pt',weights_only=True)['model'])
    runner=Runner(model,ev,x,Opt14(m,24));q,r=runner.q,runner.raw
    pipe=Pipeline(model,r,32,.25,a.mode)
    folder=ROOT/f'profile/{a.mode}';folder.mkdir(parents=True,exist_ok=True)
    def cold():pipe.refresh_history(r['kh'],r['vh']);return pipe.run(q,r['kc'],r['vc'])
    functions=[lambda:pipe.run(q,r['kc'],r['vc']),cold,runner.dense,lambda:runner.warm(.35,8)]
    graphs=[capture(fn)[0] for fn in functions]
    for graph in graphs:timing(graph,100)
    samples=[]
    for i in range(4):
        row=[None]*4
        for j in (range(4) if i%2==0 else reversed(range(4))):row[j]=timing(graphs[j],100)
        samples.append(row)
    expected=functions[0]()
    def scoped():
        scope=torch.profiler.record_function
        with scope('OPT::current_KV'):kv=pipe.cache.current(r['kc'],r['vc'])
        with scope('OPT::current_K_quantize'):pipe.quant.current()
        with scope('OPT::Q_group_gather'):ix=order(q,model.tree('q'));qq=gather(q,ix)
        with scope('OPT::route_scores'):fine,ids=scores(qq,kv.state,.35,8)
        with scope('OPT::route_select'):
            chosen=native_mask(fine,ids,296) if a.mode=='native' else select(fine,ids,296,32,.25)
        with scope('OPT::attention'):
            out=native(qq,kv,chosen,pipe.quant) if a.mode=='native' else attention(qq,kv,chosen,32,pipe.quant,2,2,2)
        with scope('OPT::restore'):return restore(out,ix)
    for _ in range(3):scoped()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):actual=scoped()
        torch.cuda.synchronize()
    assert torch.equal(actual,expected)
    f=folder/'trace.json';prof.export_chrome_trace(str(f));events=json.loads(f.read_text())['traceEvents']
    with gzip.open(folder/'trace.json.gz','wt') as stream:json.dump({'traceEvents':events},stream)
    f.unlink()
    scopes=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('OPT::')]
    launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    modules=collections.defaultdict(float);kernels={}
    for e in events:
        if e.get('cat')!='kernel':continue
        api=launches.get(e.get('args',{}).get('correlation'))
        candidates=[s for s in scopes if api and s['tid']==api['tid'] and s['ts']<=api['ts']<=s['ts']+s['dur']]
        assert candidates,e
        name=min(candidates,key=lambda s:s['dur'])['name'][5:];ms=e['dur']/5000
        modules[name]+=ms
        row=kernels.setdefault((name,e['name']),dict(module=name,kernel=e['name'],ms=0.,registers=e.get('args',{}).get('registers per thread'),occupancy=e.get('args',{}).get('est. achieved occupancy %')))
        row['ms']+=ms
    result=dict(sample=x['row'],mode=a.mode,group=32,budget=.25 if a.mode=='shared' else .15,
        timings=dict(zip(['warm_ms','cold_ms','dense_ms','opt18_warm_ms'],[statistics.median(s[i] for s in samples) for i in range(4)])),samples=samples,
        profile_module_kernel_ms=dict(modules),kernels=sorted(kernels.values(),key=lambda x:-x['ms']),exact_profile_output=True)
    atomic_json(folder/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['kernels','sample','samples']}),flush=True)
if __name__=='__main__':main()
