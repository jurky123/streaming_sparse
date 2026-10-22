"""Phase0: no optimizer updates; full36-pair real Hook/raw-QKV correctness gate."""
import argparse,json,math,os
import torch
from .common import HERE,source_identity,atomic_json,atomic_save,output,pair_index,file_sha
from .data import data,audit,validate_event
from .exact import teacher,current_stats,singleton,set_gain,retained_residual
from .sketch import Sketch
from .router import Router
from ..v05.primitives import select
from ..attention.block_statistics import exact_attention


def check(events):
    first=events[0];pair=pair_index(first['layer_id'],first['head_id']);model=Router(arms=('R3-local','R4')).cuda()
    state=model.sketch(first['k_hist'],first['v_hist'],pair);reports=[];losses=[];maximum=0.;backend=0.
    indices=torch.tensor([0,15,16,31,80,127],device='cuda')
    for event in events:
        validity=validate_event(event);q=event['q'][indices];target=event['o_dense'][indices].float()
        t=teacher(q,event);dense=t.dense();maxdense=float((dense-target).abs().max());maximum=max(maximum,maxdense)
        torch.testing.assert_close(dense,target,atol=5e-5,rtol=5e-5)
        if 'o_backend' in event:
            backend=max(backend,float((dense-event['o_backend'][indices].float()).square().sum()/dense.square().sum().clamp_min(1e-12)))
        g=t.geometry();cur=current_stats(q,event['k_current'],event['v_current'])
        identity_errors=[];gather_errors=[];literal_errors=[]
        for keep in (0.,.1,.2,.3,1.):
            budget=math.floor(keep*len(event['k_hist']));m=select(torch.randn_like(t.logz),state['costs'],budget)
            output_hard=t.output(m);D,z=retained_residual(g,m)
            expected=(-D/z[:,None]).to(output_hard.dtype)
            torch.testing.assert_close(output_hard-dense,expected,atol=5e-5,rtol=5e-5)
            identity_errors.append(float((output_hard-dense-expected).abs().max()))
            literal=-((1-m.double())[...,None]*g['_d64']).sum(1)/z[:,None]
            torch.testing.assert_close(expected,literal.to(expected.dtype),atol=5e-5,rtol=5e-5)
            literal_errors.append(float((expected.double()-literal).abs().max()))
            for i in (0,len(q)-1):
                selected=m[i].repeat_interleave(64)[:len(event['k_hist'])].bool()
                direct=exact_attention(q[i:i+1],torch.cat((event['k_hist'][selected],event['k_current'])),torch.cat((event['v_hist'][selected],event['v_current'])))
                torch.testing.assert_close(output_hard[i:i+1],direct,atol=5e-5,rtol=5e-5)
                gather_errors.append(float((output_hard[i:i+1]-direct).abs().max()))
            if ((m*state['costs']).sum(-1)>budget).any():raise AssertionError('Budget violation')
        routes=model.route(q,state,cur)
        for n,r in routes.items():
            torch.testing.assert_close(r['mask'].detach(),r['hard_mask'],atol=0,rtol=0)
            singles=model.route(q[:1],state,(cur[0][:1],cur[1][:1]))[n]
            torch.testing.assert_close(r['scores'][:1],singles['scores'],atol=5e-5,rtol=5e-5)
            losses.append((t.output(r['mask'])-target).square().mean())
        reports.append(dict(step=event['step_id'],max_dense_error=maxdense,max_gather_error=max(gather_errors),
                            max_identity_error=max(identity_errors),max_literal_identity_error=max(literal_errors),
                            validity=validity,query_tokens=event['query_indices'][indices].cpu().tolist()))
    sum(losses).backward();gradient={}
    for name,module in [('assignment',model.sketch),('base',model.base),*model.corrections.items()]:
        gs=[p.grad for p in module.parameters() if p.grad is not None]
        gradient[name]=sum(float(g.square().sum()) for g in gs)**.5
        if not gs or not all(torch.isfinite(g).all() for g in gs) or gradient[name]<=0:raise AssertionError('Nonfinite/absent gradient '+name)
    return dict(passed=True,pair=[first['layer_id'],first['head_id']],steps=reports,gradient_norms=gradient,
                max_dense_error=maximum,backend_relative_error=backend,H_policy='One historical sketch for all4 steps, no parameter updates')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(HERE/'reports/phase0.json'));a=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one GPU')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ds=data('train');hook=audit()
    if not hook['backbone']['exists']:raise FileNotFoundError(hook['backbone']['logical'])
    actual=file_sha(hook['backbone']['logical'])
    if actual!=hook['backbone']['expected_sha256']:raise ValueError('Frozen backbone SHA mismatch')
    hook['backbone']['verified_sha256']=actual
    chosen=[min(vs) for _,vs in sorted(ds.strata.items())];reports=[]
    for i,version in enumerate(chosen):
        events=ds.load(version,'cuda');torch.manual_seed(2026)
        try:report=check(events)
        except Exception as exc:
            output(HERE/'reports/phase0_failure');atomic_save(events,HERE/'reports/phase0_failure/reproduce.pt')
            atomic_json(HERE/'reports/phase0_failure/error.json',dict(source_sha256=source_identity(),snapshot=version,error=repr(exc)))
            raise
        reports.append(dict(snapshot=version,**report));print(json.dumps(dict(pair=i+1,total=36)),flush=True)
    atomic_json(a.output,dict(passed=True,source_sha256=source_identity(),data_fingerprint=ds.fingerprint(),hook=hook,
        gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),optimizer_updates=0,
        full_head_coverage=len(reports)==36,checks=reports))


if __name__=='__main__':main()
