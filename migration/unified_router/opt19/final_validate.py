"""Validate the combined optimized pipeline, including warm and cold timings."""
import argparse,json,statistics,math,gc
import torch
from ..common import config,atomic_json,sha
from ..opt18.setup import ROOT as PARENT
from ..opt17.data import Events
from ..opt17.model import JointGroupedRouter
from ..opt5.probe import capture,timing
from .pipeline import Pipeline
from .probe import ROOT,nmse

CONFIGS=[dict(name='native32',mode='native',group=32,budget=.15)]+[
    dict(name=f'shared{g}_b{round(b*100)}',mode='shared',group=g,budget=b)
    for g in [16,32] for b in [.15,.20,.25]]

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.cache_size_limit=256
    folder=ROOT/f'final/layer{a.layer:02d}_{a.variant}';folder.mkdir(parents=True,exist_ok=True)
    status=folder/'status.json';atomic_json(status,dict(status='loading',completed=0,total=96))
    _,m=config(PARENT);events=Events(m,a.layer)
    source=PARENT/f'training/layer{a.layer:02d}/{a.variant}/best.pt'
    model=JointGroupedRouter(m,a.layer,a.variant).cuda().eval()
    model.load_state_dict(torch.load(source,map_location='cuda',weights_only=True)['model'])
    prompts=sorted({x['row']['prompt_id'] for x in events.val})[:2]
    previous=json.loads((ROOT/f'validation/layer{a.layer:02d}_{a.variant}/result.json').read_text())
    rows={cfg['name']:[] for cfg in CONFIGS};timings=[];parent_rows=[]
    for index,item in enumerate(events.val):
        q,raw,truth,targets=events.gpu(item)
        old,diag=model.deployed(q,raw,targets)
        parent_rows.append(dict(sample_id=item['row']['sample_id'],nmse=nmse(old,truth)))
        pipe=Pipeline(model,raw);nh=raw['kh'].shape[1];n=q.shape[1];h=q.shape[0];nk=nh+raw['kc'].shape[1];b=diag['full_mask'].shape[-1]
        need_timing=item['row']['prompt_id'] in prompts and item['row']['step']==3 and item['row']['generation_seed']==0
        def dense():return torch.nn.functional.scaled_dot_product_attention(q[None],torch.cat((raw['kh'],raw['kc']),1)[None],torch.cat((raw['vh'],raw['vc']),1)[None])[0]
        for cfg in CONFIGS:
            pipe.mode=cfg['mode'];pipe.group=cfg['group'];pipe.budget=cfg['budget']
            out,d=pipe.run(q,raw['kc'],raw['vc'],True);out=out[:,targets].float()
            assert torch.isfinite(out).all()
            selected=d['selection'];g=cfg['group']
            if cfg['mode']=='native':
                cost=torch.full((b,),64.,device=q.device);cost[73::74]=8.
                keep=float((selected*cost).sum()/(h*n*nk))
                executed=float(torch.nn.functional.pad(selected,(0,0,0,(-n)%g)).reshape(h,-1,g,b).any(2).sum()*g*64/(h*n*nk))
            else:
                costs=torch.where(selected%74==73,8.,64.).sum(-1)
                qcost=torch.full((selected.shape[1],),g,device=q.device);qcost[-1]=n-(len(qcost)-1)*g
                keep=float((costs*qcost).sum()/(h*n*nk))
                executed=selected.numel()*g*64/(h*n*nk)
            row=dict(sample_id=item['row']['sample_id'],nmse=nmse(out,truth),nmse_vs_native=nmse(out,old),keep=keep,executed=executed)
            # Record differences against the preliminary implementation, not just a single probe.
            before=previous['rows'][cfg['name']][index]
            assert before['sample_id']==row['sample_id']
            row['nmse_delta_vs_preliminary']=row['nmse']-before['nmse']
            rows[cfg['name']].append(row)
            if need_timing:
                warm=lambda:pipe.run(q,raw['kc'],raw['vc'])
                def cold():
                    pipe.refresh_history(raw['kh'],raw['vh'])
                    return pipe.run(q,raw['kc'],raw['vc'])
                fns=[warm,cold,dense];graphs=[]
                for fn in fns:
                    graph,y=capture(fn);graph.replay();torch.cuda.synchronize()
                    if fn is not dense:assert torch.equal(y[:,targets].float(),out),(cfg,'graph output mismatch')
                    graphs.append(graph)
                for graph in graphs:timing(graph,80)
                samples=[]
                for j in range(4):
                    sample=[None]*3
                    for k in (range(3) if j%2==0 else reversed(range(3))):sample[k]=timing(graphs[k],30)
                    samples.append(sample)
                timings.append(dict(config=cfg['name'],sample_id=item['row']['sample_id'],history_tokens=nh,
                    warm_ms=statistics.median(x[0] for x in samples),cold_ms=statistics.median(x[1] for x in samples),
                    dense_ms=statistics.median(x[2] for x in samples),samples=samples))
                del graphs,graph,y
        atomic_json(status,dict(status='validating',completed=index+1,total=96))
        atomic_json(folder/'partial.json',dict(rows=rows,timings=timings,parent_rows=parent_rows))
        if index%12==11:print(a.layer,a.variant,index+1,flush=True)
        del pipe,diag,d
    summaries=[]
    for cfg in CONFIGS:
        rr=rows[cfg['name']];tt=[x for x in timings if x['config']==cfg['name']]
        assert len(rr)==96 and len(tt)==6
        summaries.append(dict(**cfg,**{k:statistics.mean(x[k] for x in rr) for k in ['nmse','nmse_vs_native','keep','executed','nmse_delta_vs_preliminary']},
            **{k:statistics.mean(x[k] for x in tt) for k in ['warm_ms','cold_ms','dense_ms']}))
    result=dict(layer=a.layer,variant=a.variant,checkpoint=str(source),checkpoint_sha256=sha(source),configs=summaries,
        rows=rows,timings=timings,parent_rows=parent_rows,scope='96 Val, fullQ4680, accuracy128 positions, six timed blocks, no video inference')
    atomic_json(folder/'result.json',result);atomic_json(status,dict(status='completed',completed=96,total=96))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
