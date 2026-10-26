import argparse,json,torch,statistics,gc
from .setup import ROOT,PARENT
from .model import QueryRouter
from .data import Events
from .pipeline import Pipeline
from ..opt20.model import TileRouter
from ..opt16.pipeline import GroupedRouter
from ..opt27.pipeline import Pipeline as Baseline
from ..opt27.prep_probe import LargeWorkspace
from ..opt5.probe import capture,timing
from ..common import atomic_json
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);p.add_argument('--variant',required=True);p.add_argument('--stage',default='pilot');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 m=json.loads((ROOT/'experiment.json').read_text());events=Events(m,a.layer);model=QueryRouter(m,a.layer,a.variant).cuda();checkpoint=ROOT/f'training/layer{a.layer:02d}/{a.variant}/best.pt';model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['model']);model.eval().requires_grad_(False);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
 base=TileRouter(m,a.layer,'Tile20').cuda().eval().requires_grad_(False);base.load_state_dict(torch.load(PARENT/f'training/layer{a.layer:02d}/Tile20/best.pt',map_location='cuda',weights_only=True)['model']);base.pipeline=GroupedRouter(base.dictionary(),base.tree('q'),base.tree('k'),compile=False)
 pool=[x for x in events.val if x['row']['step']==3];items=[]
 for nh in [4680,9360,14040]:items += [x for x in pool if x['row']['history_tokens']==nh][:2]
 rows=[]
 for item in items:
  q,raw,truth,targets=events.gpu(item);prev=events.previous(item,model.options['refresh']);old=Baseline(base,raw,True,LargeWorkspace(raw));new=Pipeline(model,raw);bg,bo=capture(lambda:old.run(q,raw['kc'],raw['vc']));modes=['fresh']if prev is None else['fresh','reuse'];times={}
  for mode in modes:
   new.refresh_assignment=mode=='fresh'
   if mode=='reuse':new.run(q,raw['kc'],raw['vc'],previous_q=prev)
   fn=lambda:new.run(q,raw['kc'],raw['vc']);expected=fn();ng,no=capture(fn);ng.replay();torch.cuda.synchronize();assert torch.equal(expected,no)
   reference,_=model.deployed(q,raw,targets,prev if mode=='reuse'else None);assert torch.equal(no[:,targets].float(),reference)
   timing(ng,100);pairs=[]
   for i in range(5):
    if i%2:before=timing(bg,30);after=timing(ng,30)
    else:after=timing(ng,30);before=timing(bg,30)
    pairs.append([before,after])
   times[mode]=dict(baseline_ms=statistics.median(x[0]for x in pairs),ms=statistics.median(x[1]for x in pairs),samples=pairs)
   del ng,no,reference
  # Pair and early policies both refresh on two out of the four native steps.
  ms=statistics.mean(x['ms']for x in times.values());bs=statistics.mean(x['baseline_ms']for x in times.values());rows.append(dict(sample_id=item['row']['sample_id'],history_tokens=item['row']['history_tokens'],modes=times,ms=ms,baseline_ms=bs,speedup=bs/ms))
  print(rows[-1],flush=True);del bg,bo,new,old,q,raw,prev;gc.collect();torch.cuda.empty_cache()
 result=dict(completed=True,layer=a.layer,variant=a.variant,stage=a.stage,rows=rows,ms=statistics.mean(x['ms']for x in rows),baseline_ms=statistics.mean(x['baseline_ms']for x in rows),scope='Warm sparse module, cached history; current KV,Q grouping,aggregation,coarse/fine,selection,native attention,restore. Pair/early average measured fresh+reuse equally; not video wall time.',best_validation=json.loads(checkpoint.with_name('best_validation.json').read_text()))
 result['speedup']=result['baseline_ms']/result['ms'];atomic_json(ROOT/f'bench/{a.stage}/layer{a.layer:02d}/{a.variant}.json',result)
if __name__=='__main__':main()
