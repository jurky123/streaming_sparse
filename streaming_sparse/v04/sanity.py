"""Real/synthetic v0.4 correctness only: zero optimizer updates, no capture."""
import argparse
import os
from pathlib import Path
import torch
from .data import SnapshotData
from .identity import source_identity
from .model import PairedRouter,RouterConfig
from ..v03.evaluate import statistics
from ..v03.sanity import synthetic_events
from ..v03.metrics import error_parts
from ..attention.block_statistics import exact_attention
from ..capture.dataset_io import query_slices
from ..v02.evaluation_journal import atomic_json
from .paths import protect_output


def check(model,events):
    dtypes={}
    def capture(name):
        def hook(module,args,output):dtypes[name]=str(output.dtype)
        return hook
    handles=[b.fusion.base.register_forward_hook(capture(name)) for name,b in model.branches.items()]
    first=events[0]
    state=model.precompute(first['k_hist'],first['v_hist'],head_id=first['head_id'],layer_id=first['layer_id'])
    cache=state['contexts']['B'].detach().clone()
    losses=[]
    scores={name:[] for name in ('A','B')}
    for event in events:
        selection=query_slices(event,16)[0]
        q,target=event['q'][selection],event['o_dense'][selection]
        stat=statistics(q,event,state['assignment'])
        torch.testing.assert_close(stat.dense(),target.float(),atol=5e-5,rtol=5e-5)
        routes=model.route(q,state,.2)
        for name,route in routes.items():
            selected=state['hard']@route['hard_mask']>0
            direct=exact_attention(q,torch.cat((event['k_hist'][selected],event['k_current'])),
                                   torch.cat((event['v_hist'][selected],event['v_current'])))
            torch.testing.assert_close(stat.output(route['mask']),direct,atol=5e-5,rtol=5e-5)
            se,en=error_parts(stat.output(route['mask']),target)
            losses.append(se/(en+1e-8))
            scores[name].append(route['scores'].detach())
    loss=torch.stack(losses).mean()
    pg,=torch.autograd.grad(loss,state['probabilities'],retain_graph=True)
    if not torch.isfinite(pg).all() or float(pg.norm())==0:
        raise AssertionError('Disconnected clustering gradient')
    loss.backward()
    norms={name:sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5
           for name,module in [('frontend',model.frontend),('A',model.branches['A']),('B',model.branches['B'])]}
    if any(not torch.isfinite(torch.tensor(v)) or v<=0 for v in norms.values()):
        raise AssertionError(norms)
    torch.testing.assert_close(cache,state['contexts']['B'].detach(),rtol=0,atol=0)
    changed={name:any(float((a-b).abs().max())>1e-6 for a,b in zip(v,v[1:])) for name,v in scores.items()}
    if not all(changed.values()):raise AssertionError('Query-insensitive score path')
    for handle in handles:handle.remove()
    return dict(passed=True,partition_hash_A=state['partition_hash'],partition_hash_B=state['partition_hash'],
                shared_z=True,dense_and_raw_gather=True,gradient_norms=norms,cluster_gradient_norm=float(pg.norm()),
                B_query_independent_context_unchanged=True,score_changed_with_query=changed,steps=len(events),
                fusion_compute_dtype=dtypes,frontend_dtype=str(state['z'].dtype))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic',action='store_true');mode.add_argument('--data')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--full-head-coverage',action='store_true')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.full_head_coverage and not args.data:raise ValueError('Full head check requires real QKV')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Expose one CUDA GPU')
    selected=[]
    data=None
    if args.data:
        data=SnapshotData(args.data,'train',require_complete=args.full_head_coverage)
        for key in sorted(data.strata):
            selected.append(sorted(data.strata[key])[0])
        if not args.full_head_coverage:selected=selected[:1]
    else:selected=['synthetic']
    reports=[]
    for version in selected:
        torch.manual_seed(0)
        model=PairedRouter(RouterConfig(target_size=16 if not args.data else 64)).to(args.device)
        events=data.load(version,args.device) if data else synthetic_events(args.device)
        reports.append(dict(cache_version=version,**check(model,events)))
    target=protect_output(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(target,dict(passed=True,version='0.4',real_qkv=bool(args.data),
        full_head_coverage=args.full_head_coverage,
        bf16_checked=all(set(r['fusion_compute_dtype'].values())=={'torch.bfloat16'} for r in reports),
        data_fingerprint=data.fingerprint() if data else None,
        gpu_name=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=str(torch.__version__),
        source_sha256=source_identity(),optimizer_updates=0,device=args.device,capacity=model.capacity(),checks=reports))


if __name__=='__main__':main()
