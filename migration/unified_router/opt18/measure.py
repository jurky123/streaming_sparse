"""Full-Q block timings with explicit cache boundaries and paired references."""
from types import SimpleNamespace
from pathlib import Path
import statistics
import torch
from ..core import build
from ..fast_route import prepare
from ..learning import ActionHead
from ..opt14.network import Top3Route
from ..opt9.segmented_attention import attention
from ..opt16.grouping import order,gather,restore
from ..opt16.history import HistoryBuilder
from ..opt17.route import route
from ..opt5.probe import capture,timing

class Opt14:
    def __init__(self,m,layer):
        self.layer=layer
        self.parameters=torch.load(Path(m['parent_experiment'])/f'action_basis/layer{layer:02d}.pt',map_location='cuda',weights_only=True)['parameters']
        head_state=torch.load(Path(m['parent_opt11'])/f'strong_training/layer{layer:02d}/Compact/best.pt',map_location='cuda',weights_only=True)['head']
        head=ActionHead(head_state).cuda().eval();head.load_state_dict(head_state);head.requires_grad_(False)
        self.router=SimpleNamespace(kind='K32-Set',beta=m['monitor_beta'],heads={f'{layer}_K32-Set':head})
        self.route=Top3Route(batch=4680);self.build=torch.compile(build,fullgraph=True);self.prepare=torch.compile(prepare,fullgraph=True)

    def prepared(self,q,raw,step):
        s=self.build(raw['kh'],raw['vh'],raw['kc'],raw['vc'],self.parameters,need_value=False)
        p=self.prepare(s,'K32-Set');state=self.route.begin(p,s);st=q.new_tensor(step,dtype=torch.float32)
        def run():
            mask=self.route.run(self.layer,q,s,state,st,self.router)
            return attention(q,raw['kh'],raw['vh'],raw['kc'],raw['vc'],mask,pack_v=True,fuse_pack=True)
        return run

class Runner:
    def __init__(self,model,events,item,baseline):
        self.model=model;self.item=item
        self.q,self.raw,self.truth,self.targets=events.gpu(item)
        _,d=model.deployed(self.q,self.raw,self.targets)
        self.pipeline=model.pipeline;self.kv=d['kv'];self.builder=HistoryBuilder(self.pipeline)
        self.history=self.builder.history(self.raw['kh'],self.raw['vh'])
        self.old=baseline.prepared(self.q,self.raw,item['row']['step'])

    def execute(self,kv,fraction,reps):
        ix=order(self.q,self.model.tree('q'));q=gather(self.q,ix)
        mask=route(q,kv.state,self.model.keep,fraction,representatives=reps)
        nh=kv.history_tokens
        y=attention(q,kv.k[:,:nh],kv.v[:,:nh],kv.k[:,nh:],kv.v[:,nh:],mask,pack_v=True,fuse_pack=True)
        return restore(y,ix)

    def cold(self,fraction,reps):
        r=self.raw;k=torch.cat((r['kh'],r['kc']),1);v=torch.cat((r['vh'],r['vc']),1)
        return self.execute(self.pipeline.pack_kv(k,v,r['kh'].shape[1]),fraction,reps)

    def warm(self,fraction,reps):
        return self.execute(self.builder.current(self.history,self.raw['kc'],self.raw['vc']),fraction,reps)

    def dense(self):
        r=self.raw
        return torch.nn.functional.scaled_dot_product_attention(self.q[None],torch.cat((r['kh'],r['kc']),1)[None],torch.cat((r['vh'],r['vc']),1)[None])[0]

    @torch.no_grad()
    def measure(self,fraction,reps):
        expected,d=self.model.deployed(self.q,self.raw,self.targets,fraction,reps)
        warm=self.warm(fraction,reps)[:,self.targets].float()
        cold=self.cold(fraction,reps)[:,self.targets].float()
        nmse=lambda x:float((x-expected).square().sum()/expected.square().sum().clamp_min(1e-20))
        ew,ec=nmse(warm),nmse(cold)
        assert max(ew,ec)<1e-4,('timed vs evaluated kernel mismatch',ew,ec)
        fns=[lambda:self.warm(fraction,reps),lambda:self.cold(fraction,reps),self.dense,self.old]
        graphs=[capture(fn)[0]for fn in fns]
        # Warm all paths before collecting balanced event-timed replay samples.
        for g in graphs:timing(g,100)
        samples=[]
        for iteration in range(4):
            row=[None]*4
            for i in (range(4) if iteration%2==0 else reversed(range(4))):row[i]=timing(graphs[i],30)
            samples.append(row)
        return dict(sample_id=self.item['row']['sample_id'],history_tokens=self.raw['kh'].shape[1],
            warm_ms=statistics.median(x[0]for x in samples),cold_ms=statistics.median(x[1]for x in samples),
            dense_concat_ms=statistics.median(x[2]for x in samples),opt14_prepared_route_attention_ms=statistics.median(x[3]for x in samples),
            timed_warm_vs_evaluated_nmse=ew,timed_cold_vs_evaluated_nmse=ec,samples=samples)

    @torch.no_grad()
    def grouping_time(self):
        graph,_=capture(lambda:order(self.q,self.model.tree('q')));timing(graph,100)
        return statistics.median(timing(graph,100)for _ in range(3))
