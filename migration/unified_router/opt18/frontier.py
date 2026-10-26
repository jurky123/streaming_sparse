import argparse,gc,json,os,time,traceback
from pathlib import Path
import torch
from .setup import ROOT
from ..common import config,atomic_json,save,update,sha
from ..opt17.data import Events
from ..opt17.model import JointGroupedRouter
from ..opt17.train import evaluate,summary
from ..opt16.grouping import permutation
from .measure import Opt14,Runner

def timing_items(events):
    prompts=sorted({x['row']['prompt_id']for x in events.val})[:2]
    return [next(x for x in events.val if x['row']['prompt_id']==p and x['row']['history_tokens']==n and x['row']['step']==3)for p in prompts for n in [4680,9360,14040]]

@torch.no_grad()
def baseline_quality(baseline,events):
    rows=[]
    for item in events.val:
        q,raw,truth,targets=events.gpu(item);out=baseline.prepared(q,raw,item['row']['step'])()[:,targets].float()
        sse=(out-truth).square().sum();energy=truth.square().sum()
        rows.append(dict(sample_id=item['row']['sample_id'],nmse=float(sse/energy),sse=float(sse),energy=float(energy)))
    return dict(nmse=sum(x['nmse']for x in rows)/len(rows),rows=rows)

@torch.no_grad()
def run(m,layer,variant,rank):
    folder=ROOT/f'frontier/layer{layer:02d}/{variant}';folder.mkdir(parents=True,exist_ok=True)
    status=ROOT/f'frontier/rank{rank}.json'
    update(status,status='loading',layer=layer,variant=variant,completed=0)
    events=Events(m,layer);items=timing_items(events);assert len(items)==6
    source=ROOT/f'training/layer{layer:02d}/{variant}/best.pt'
    state=torch.load(source,map_location='cuda',weights_only=True)
    model=JointGroupedRouter(m,layer,variant).cuda();model.load_state_dict(state['model'])
    baseline=Opt14(m,layer)
    if not (folder/'opt14_validation.json').exists():atomic_json(folder/'opt14_validation.json',baseline_quality(baseline,events))
    # Check stable counting permutation at the largest new bucket count.
    for kind in ['random','constant']:
        labels=torch.randint(256,(2,4680),device='cuda',dtype=torch.int32)if kind=='random'else torch.zeros((2,4680),device='cuda',dtype=torch.int32)
        assert torch.equal(permutation(labels,256).long(),labels.argsort(dim=-1,stable=True))
    configs=[dict(tree='best64',depth=6,fraction=f,reps=r)for f in m['sweep_fractions']for r in (m['sweep_representatives']if f<1 else[16])]
    if variant=='Quality50':configs += [dict(tree=f'calibrated{2**depth}',depth=depth,fraction=f,reps=16)for depth in m['sweep_new_depths']for f in m['sweep_new_tree_fractions']]
    by_tree={}
    for c in configs:by_tree.setdefault(c['tree'],[]).append(c)
    results=[]
    for name,group in by_tree.items():
        depth=group[0]['depth']
        if name!='best64':
            mm=dict(m,tree_depth=depth);model=JointGroupedRouter(mm,layer,variant).cuda()
            model.load_state_dict({k:v for k,v in state['model'].items()if '_tree_'not in k},strict=False)
            q,k,ids=events.calibration_tensors(18000+layer,m['calibration_events'])
            model.set_trees(*model.propose_trees(q,k,projected=True))
            save(folder/(name+'.pt'),dict(trees={k:v for k,v in model.state_dict().items()if '_tree_'in k},calibration_events=ids,split='train',depth=depth,parent_checkpoint_sha256=sha(source)))
            del q,k
        runners=[Runner(model,events,item,baseline)for item in items]
        group_ms=sum(r.grouping_time()for r in runners)/len(runners)
        for c in group:
            key=f'{name}_c{round(c["fraction"]*100):03d}_r{c["reps"]:02d}';dest=folder/'configs'/(key+'.json')
            if dest.exists():results.append(json.loads(dest.read_text()));continue
            update(status,status='accuracy',layer=layer,variant=variant,config=key,completed=len(results),total=len(configs))
            quality=evaluate(model,events,fraction=c['fraction'],representatives=c['reps'])
            update(status,status='timing',layer=layer,variant=variant,config=key,completed=len(results),total=len(configs))
            timing_rows=[r.measure(c['fraction'],c['reps'])for r in runners]
            means={k:sum(row[k]for row in timing_rows)/len(timing_rows)for k in ['warm_ms','cold_ms','dense_concat_ms','opt14_prepared_route_attention_ms']}
            record=dict(key=key,layer=layer,variant=variant,**c,**summary(quality),**means,q_grouping_ms=group_ms,
                checkpoint=str(source),checkpoint_sha256=sha(source),timing_events=timing_rows,validation_rows=quality['rows'])
            atomic_json(dest,record);results.append(record)
            atomic_json(folder/'results.json',dict(layer=layer,variant=variant,configs=results))
            print('frontier',layer,variant,key,summary(quality),means,flush=True)
        del runners;gc.collect();torch.cuda.empty_cache()
    atomic_json(folder/'results.json',dict(layer=layer,variant=variant,configs=results))
    atomic_json(folder/'completed.json',dict(layer=layer,variant=variant,configurations=len(results),checkpoint=str(source),checkpoint_sha256=sha(source)))
    update(status,status='completed',layer=layer,variant=variant,completed=len(results),total=len(configs))

def main():
    p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant',required=True);a=p.parse_args()
    _,m=config(ROOT);rank=int(os.environ.get('RANK',0));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.cache_size_limit=256;torch._dynamo.config.accumulated_cache_size_limit=2048
    try:run(m,a.layer,a.variant,rank)
    except Exception as e:
        path=ROOT/f'frontier/rank{rank}.json';old=json.loads(path.read_text())if path.exists()else{};old.pop('updated_at',None)
        update(path,**dict(old,status='failed',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
