import json,torch,gc
from .setup import ROOT,source_step
from .model import QueryRouter
from .data import Events
from .pipeline import Pipeline
from ..opt16.pipeline import GroupedRouter
from ..opt5.probe import capture
from ..common import atomic_json
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((ROOT/'experiment.json').read_text());events=Events(m,24);rows=[]
 owners=[]
 for item in events.val:
  owner=(item['row']['trajectory_id'],item['row']['chunk'])
  if owner not in owners:owners.append(owner)
 for name in ['q64_a8_pair','q64_a8_early','q64_a16_pair']:
  model=QueryRouter(m,24,name).cuda().eval().requires_grad_(False);model.pipeline=GroupedRouter(model.dictionary(),model.tree('q'),model.tree('k'),compile=False)
  # Two different videos/chunks, then repeat the first owner through the same pipe.
  pipe=None;graphs={};inputs=None;first_outputs=[]
  for visit,owner in enumerate([owners[0],next(x for x in owners if x[0]!=owners[0][0]),owners[0]]):
   items=sorted([x for x in events.val if(x['row']['trajectory_id'],x['row']['chunk'])==owner],key=lambda x:x['row']['step']);assert len(items)==4
   for item in items:
    q,raw,truth,targets=events.gpu(item);step=item['row']['step'];prev=events.previous(item,model.options['refresh']);reference,_=model.deployed(q,raw,targets,prev)
    if pipe is None:
     pipe=Pipeline(model,raw);inputs=(torch.empty_like(q),torch.empty_like(raw['kc']),torch.empty_like(raw['vc']))
    if step==0:pipe.refresh_history(raw['kh'],raw['vh'])
    for dst,src in zip(inputs,[q,raw['kc'],raw['vc']]):dst.copy_(src)
    fresh=source_step(step,model.options['refresh'])==step;pipe.refresh_assignment=fresh
    if fresh not in graphs:graphs[fresh]=capture(lambda:pipe.run(*inputs))
    graph,out=graphs[fresh];graph.replay();torch.cuda.synchronize();assert torch.equal(out[:,targets].float(),reference),(name,visit,step,'wrong source-step A or token alignment')
    if visit==0:first_outputs.append(out.clone())
    if visit==2:assert torch.equal(out,first_outputs[step]),'owner leakage'
    rows.append(dict(variant=name,visit=visit,step=step,source_step=source_step(step,model.options['refresh']),owner=list(owner),exact=True))
   print(name,'owner',visit,'passed',flush=True)
  del model,pipe,graphs,inputs,first_outputs;gc.collect();torch.cuda.empty_cache()
 atomic_json(ROOT/'sequence_check.json',dict(passed=True,rows=rows,source_step_assignment_exact=True,original_token_alignment_exact=True,cross_owner_reset_exact=True))
if __name__=='__main__':main()
