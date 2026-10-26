import torch
from .setup import ROOT
from .model import TileRouter
from .network import TileGraph
from ..common import config,atomic_json
from ..opt17.data import Events
from ..opt16.pipeline import GroupedRouter

@torch.no_grad()
def main():
    _,m=config(ROOT);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ev=Events(m,24);items=[x for x in ev.train if x['row']['history_tokens']==14040][::8][:3]
    results={}
    for variant in m['variants']:
        model=TileRouter(m,24,variant).cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(ROOT/'initialization/layer24.pt',map_location='cuda',weights_only=True)['model'])
        graph=None;checks=[]
        for i,item in enumerate(items+items[:1]):
            q,raw,_,targets=ev.gpu(item);expected,_=model.deployed(q,raw,targets)
            if graph is None:graph=TileGraph(model,q,raw,torch.cuda.graph_pool_handle())
            out,_=graph(q,raw,('different-history',i))
            assert torch.equal(out[:,targets].float(),expected),(variant,i,'stale graph/cache')
            out2,_=graph(q,raw,('different-history',i))
            assert torch.equal(out2[:,targets].float(),expected)
            checks.append(dict(sample=item['row']['sample_id'],bit_identical=True))
        results[variant]=checks
    atomic_json(ROOT/'graph_preflight.json',dict(passed=True,changed_history_updates_static_prototypes=True,variants=results))
    print('GRAPH PREFLIGHT PASSED',flush=True)

if __name__=='__main__':main()
