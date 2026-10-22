"""Zero-update real full-head gate for independent queries, masks, caches and signs."""
import argparse
import json
import os
import torch
from .common import DEFAULT_DATA,SnapshotData,source_identity,atomic_json,protect_output
from .model import Router,RouterConfig
from .primitives import statistics
from ..capture.dataset_io import query_slices
from ..attention.block_statistics import exact_attention


def check(model,events):
    first=events[0];calls=[]
    compute_dtypes={}
    def capture(name):
        def hook(module,args,value):compute_dtypes[name]=str(value.dtype)
        return hook
    dtype_handles=[module.register_forward_hook(capture(name)) for name,module in (
        ('pre_q',model.frontend.pre.qkv.base),
        ('fusion',model.query_fusion.fusion.linears[0].base),
        ('ordinary',model.branches['R2'].interaction[0].qkv.base),
        ('signed_vector',model.branches['R3'].interaction[0].u.linears[0].base))]
    handle=model.frontend.register_forward_hook(lambda *args:calls.append(1))
    state=model.precompute(first['k_hist'],first['v_hist'],head_id=first['head_id'],layer_id=first['layer_id'])
    saved_h=state['h'].detach().clone();losses=[];signs=[];dtypes={}
    for event in events:
        selection=query_slices(event,16)[0]
        q=event['q'][selection];target=event['o_dense'][selection]
        stat=statistics(q,event,state['assignment'])
        torch.testing.assert_close(stat.dense(),target.float(),rtol=5e-5,atol=5e-5)
        routes=model.route(q,state,.2,trace=True)
        for route in routes.values():
            if route['fusion'].data_ptr()!=routes['R2']['fusion'].data_ptr() or route['query_embedding'].data_ptr()!=routes['R2']['query_embedding'].data_ptr():
                raise AssertionError('Query representation/fusion is not exactly shared')
        changed=q.clone();changed[0]=changed[0]+torch.linspace(-1,1,128,device=q.device)
        altered=model.predict(changed,state)
        for name,route in routes.items():
            if route['scores'].shape!=(16,len(state['h'])) or route['mask'].shape!=route['scores'].shape:
                raise AssertionError('Query axis collapsed')
            torch.testing.assert_close(route['scores'][1:],altered[name]['scores'][1:],rtol=0,atol=0)
            if float((route['scores'][0]-altered[name]['scores'][0]).abs().max())<=1e-8:
                raise AssertionError('Query projection/fusion inactive')
            retained=(route['hard_mask']*state['hard'].sum(0)).sum(-1)
            if bool((retained>route['budget']).any()):raise AssertionError('Per-query budget exceeded')
            for index in (0,15):
                tokens=state['hard']@route['hard_mask'][index]>0
                direct=exact_attention(q[index:index+1],torch.cat((event['k_hist'][tokens],event['k_current'])),
                                       torch.cat((event['v_hist'][tokens],event['v_current'])))
                torch.testing.assert_close(stat.output(route['mask'])[index:index+1],direct,rtol=5e-5,atol=5e-5)
            losses.append((stat.output(route['mask'])-target.float()).square().sum()/(target.float().square().sum()+1e-8))
            dtypes[name]=dict(fusion=str(route['fusion'].dtype),score=str(route['scores'].dtype))
        for layer in routes['R3']['layers']:
            r=layer['relation'];signs.append(dict(positive=bool((r>1e-4).any()),negative=bool((r< -1e-4).any()),dtype=str(r.dtype)))
            if r.dtype!=torch.float32 or float(r.diagonal(dim1=-1,dim2=-2).abs().max())!=0:
                raise AssertionError('Signed geometry precision/self-mask error')
        zero=model.predict(q,state,intervention='zero',only=('R3',))['R3']
        torch.testing.assert_close(zero['correction'],torch.zeros_like(zero['correction']),rtol=0,atol=0)
    loss=torch.stack(losses).mean()
    pg,=torch.autograd.grad(loss,state['probabilities'],retain_graph=True)
    if not torch.isfinite(pg).all() or float(pg.norm())==0:raise AssertionError('Clustering gradient inactive')
    loss.backward()
    modules=[('frontend',model.frontend),('shared_query_fusion',model.query_fusion),*model.branches.items()]
    modules.extend((f'R3_u{index}',block.u) for index,block in enumerate(model.branches['R3'].interaction))
    norms={}
    for name,module in modules:
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise AssertionError('Nonfinite/missing gradient: '+name)
        norms[name]=sum(float(g.square().sum()) for g in gradients)**.5
        if norms[name]<=0:raise AssertionError('Zero gradient: '+name)
    torch.testing.assert_close(saved_h,state['h'].detach(),rtol=0,atol=0)
    if calls!=[1] or not all(s['positive'] and s['negative'] for s in signs):raise AssertionError('Cache/sign gate failed')
    if compute_dtypes!={'pre_q':'torch.bfloat16','fusion':'torch.bfloat16','ordinary':'torch.bfloat16','signed_vector':'torch.float32'}:
        raise AssertionError(('Autocast/FP32 island mismatch',compute_dtypes))
    handle.remove()
    for hook in dtype_handles:hook.remove()
    return dict(passed=True,shape=list(routes['R3']['scores'].shape),independent_query_perturbation=True,
                dense_and_raw_gather=True,per_query_budget=True,H_precompute_calls=len(calls),four_step_H_reuse=True,
                clustering_gradient_norm=float(pg.norm()),gradient_norms=norms,signed_relations=signs,
                zero_message_zero_correction=True,shared_query_representation=True,dtypes=dtypes,
                actual_linear_compute_dtypes=compute_dtypes,partition_hash=state['partition_hash'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default=str(DEFAULT_DATA));parser.add_argument('--output',required=True)
    parser.add_argument('--single',action='store_true');args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one CUDA GPU')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=SnapshotData(args.data,'train')
    selected=[min(values) for key,values in sorted(data.strata.items())]
    if args.single:selected=selected[:1]
    reports=[]
    for i,version in enumerate(selected):
        torch.manual_seed(0);model=Router().cuda()
        events=data.load(version,'cuda')
        reports.append(dict(cache_version=version,layer=events[0]['layer_id'],head=events[0]['head_id'],**check(model,events)))
        print(json.dumps(dict(completed=i+1,total=len(selected),layer=events[0]['layer_id'],head=events[0]['head_id'])),flush=True)
        del model,events
    target=protect_output(args.output);target.parent.mkdir(exist_ok=True,parents=True)
    atomic_json(target,dict(passed=True,real_qkv=True,full_head_coverage=len(reports)==36,source_sha256=source_identity(),
               data_fingerprint=data.fingerprint(),gpu_name=torch.cuda.get_device_name(0),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=str(torch.__version__),
               optimizer_updates=0,checks=reports))


if __name__=='__main__':main()
