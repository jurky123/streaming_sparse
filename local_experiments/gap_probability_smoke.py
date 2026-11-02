"""Two early-prefix bit-exact regressions; no redundant VAE/LPIPS if latents match."""
import json,torch
from pathlib import Path
from .broad_common import Engine,ROOT as DATA
from .gap_probability_runtime import GapProbabilityRuntime
from .rollout_suite import generate_chunk
from .learned_cache.io import atomic,digest,RunLock,disk_guard
ROOT=Path('artifacts/gap_route_stats_20261031/probability_smoke')
@torch.no_grad()
def main():
    lock=RunLock(ROOT);rt=None
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=Path('artifacts/four_gaps_20261031/shared/epoch1.pt')
        ident=dict(model=digest(model),code={n:digest(Path('local_experiments')/n) for n in ['gap_probability_smoke.py','gap_probability_runtime.py','gap_probability_stats.py','gap_spatial_softmax.py']})
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==ident
        else:atomic(ROOT/'identity.json',ident)
        engine=Engine();rt=GapProbabilityRuntime(engine,'artifacts/full_route_grad_20261031/epoch1.pt',probes=16)
        rt.model.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['model'])
        for name,seed in [('traffic_crossing',404),('bird_flight',505)]:
            dest=ROOT/f'{name}.json'
            if dest.exists():continue
            disk_guard(ROOT,50,1);rt.reset();atomic(ROOT/'status.json',dict(status='running',case=name))
            emb=torch.load(DATA/'embeddings'/f'{name}.pt',weights_only=True).cuda()
            refpath=Path('artifacts/gap_vector_flat_20261031/quality')/f'test_{name}_s{seed}'/'flat.latents.pt'
            ref=torch.load(refpath,weights_only=True)
            zs=[generate_chunk(engine,emb,seed,c) for c in range(9)];z=torch.cat(zs,1).cpu()
            exact=torch.equal(z,ref);row=dict(case=name,seed=seed,reference=str(refpath),full_prefix_latents_exact=exact,max_error=float((z.float()-ref.float()).abs().max()),runtime=rt.finish())
            atomic(dest,row);print(json.dumps({k:v for k,v in row.items() if k!='runtime'}),flush=True)
            assert exact,'Prefix differs: requires RGB quality measurement before acceptance'
        atomic(ROOT/'status.json',dict(status='complete',passed=True,scope='Two nine-chunk prefixes bit-exact to previously evaluated flat baseline; no new broad quality claim.'))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if rt is not None:rt.close()
        lock.close()
if __name__=='__main__':main()
