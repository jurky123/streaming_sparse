"""P0 numerical and gradient acceptance checks; optional real capture audit."""
import argparse
import json
from pathlib import Path
import torch
from ..router.model import SparseRouter, RouterConfig
from ..router.budget_selection import soft_budget_relaxation, select_under_budget
from ..attention.block_statistics import differentiable_block_stats, exact_attention
from ..attention.reconstruction_loss import normalized_attention_mse
from ..capture.dataset_io import AttentionDataset
from ..train.train_router import gradient_norms, to_device


def check_sample(sample, model):
    q = sample['q'][:model.config.query_tile]
    kh, vh, kc, vc = [sample[k] for k in ('k_hist','v_hist','k_current','v_current')]
    state = model.precompute(kh, vh, sample['token_positions'], sample.get('head_id',0), sample.get('layer_id',0))
    a, hard = state['assignment'], state['hard']
    if not torch.equal(hard.sum(-1), torch.ones(len(kh), device=kh.device)):
        raise AssertionError('History token not assigned exactly once')
    if (hard.sum(0) == 0).any():
        raise AssertionError('Unexpected empty cluster after repair')
    stats = differentiable_block_stats(q, kh, vh, a, kc, vc, key_chunk=73)
    dense = exact_attention(q, torch.cat((kh,kc)), torch.cat((vh,vc)))
    torch.testing.assert_close(stats.dense(), dense, rtol=3e-5, atol=3e-5)
    errors = []
    loss = 0
    for keep in [0, .1, .2, .3, 1]:
        route = model.route(q, state, keep)
        counts = hard.sum(0)
        kept = int((counts * route['hard_mask']).sum())
        if kept > route['budget']:
            raise AssertionError('Budget exceeded')
        selected = (hard @ route['hard_mask']) > 0
        direct = exact_attention(q, torch.cat((kh[selected],kc)), torch.cat((vh[selected],vc)))
        reconstructed = stats.output(route['mask'])
        torch.testing.assert_close(reconstructed, direct, rtol=5e-5, atol=5e-5)
        errors.append(float(normalized_attention_mse(reconstructed, direct)))
        if keep == .3:
            loss = normalized_attention_mse(reconstructed, dense)
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = gradient_norms(model)
    if any(not 0 < value < float('inf') for value in gradients.values()):
        raise AssertionError(f'Missing/nonfinite attention-loss gradient: {gradients}')
    backend_error = None
    if 'o_backend' in sample:
        backend_error = float(normalized_attention_mse(dense, sample['o_backend'][:len(q)]))
        if backend_error > 1e-4:
            raise AssertionError(f'FP32/backend relative error too large: {backend_error}')
        torch.testing.assert_close(dense, sample['o_dense'][:len(q)], rtol=3e-5, atol=3e-5)
    return dict(max_hard_stats_relative_error=max(errors), gradients=gradients,
                dense_backend_error=backend_error, blocks=hard.shape[1],
                block_min=int(hard.sum(0).min()), block_max=int(hard.sum(0).max()))


def cache_check(device):
    from types import SimpleNamespace
    from wan.modules.causal_model import CausalWanSelfAttention
    from wan.modules.model import rope_params
    from ..integration.self_forcing_adapter import SelfForcingAdapter
    from ..capture.capture_runner import sdpa_backend
    attention = CausalWanSelfAttention(128, 1, local_attn_size=6, sink_size=1).to(device=device, dtype=torch.bfloat16).eval()
    cache = dict(k=torch.zeros((1,24,1,128),device=device,dtype=torch.bfloat16),
                 v=torch.zeros((1,24,1,128),device=device,dtype=torch.bfloat16),
                 global_end_index=torch.tensor([0],device=device),local_end_index=torch.tensor([0],device=device))
    captures = []
    def callback(context, q, k, v, output):
        captures.append((context['positions'].clone(), k[:, :context['history_tokens']].clone(), context['phase']))
        dense = exact_attention(q[0,:,0],k[0,:,0],v[0,:,0])
        assert float(normalized_attention_mse(dense, output[0,:,0])) < 1e-4
    adapter = SelfForcingAdapter(SimpleNamespace(blocks=[SimpleNamespace(self_attn=attention)]), capture=callback)
    with torch.no_grad(), sdpa_backend(), adapter:
        for chunk in range(3):
            for step in range(3):
                adapter.event = dict(prompt_id='fixture',chunk_id=chunk,step_id=step,timestep=500 if step<2 else 0,
                                     phase='denoise' if step<2 else 'context')
                x = torch.randn((1,12,128),device=device,dtype=torch.bfloat16)
                attention(x,torch.tensor([12],device=device),torch.tensor([[3,2,2]],device=device),
                          rope_params(32,128).to(device),None,kv_cache=cache,current_start=chunk*12)
    expected = torch.tensor(list(range(4))+list(range(16,36)),device=device)
    actual = captures[-1][0]
    actual = actual[:,0]*4+actual[:,1]*2+actual[:,2]
    assert torch.equal(actual,expected), (actual,expected)
    # History is stable across denoising steps, but includes a newly committed clean chunk later.
    assert torch.equal(captures[3][1],captures[4][1])
    assert torch.equal(captures[6][1],captures[7][1])
    result = {'rolled_absolute_positions': actual.cpu().tolist(), 'history_reused_across_steps': True}
    # Exercise actual adapter gather, H reuse and dense context refresh with the
    # same real attention module, independently checking every selected tile.
    router = SparseRouter(RouterConfig(target_size=4,query_tile=8)).to(device).eval()
    rebuilds = []
    original_precompute = router.precompute
    def counted(*args, **kwargs):
        rebuilds.append(1)
        return original_precompute(*args,**kwargs)
    router.precompute = counted
    class CheckedAdapter(SelfForcingAdapter):
        def _sparse(self, context, q, k, v):
            output = super()._sparse(context,q,k,v)
            nh = context['history_tokens']
            if nh:
                state = self.encoded[(0,0)][1]
                for start in range(0,q.shape[1],8):
                    tile = q[0,start:start+8,0]
                    route = self.router.route(tile,state,self.keep_ratio)
                    stats = differentiable_block_stats(tile,k[0,:nh,0],v[0,:nh,0],state['hard'],k[0,nh:,0],v[0,nh:,0])
                    error = normalized_attention_mse(stats.output(route['hard_mask']),output[0,start:start+len(tile),0])
                    assert float(error)<1e-4
            return output
    cache['k'].zero_()
    cache['v'].zero_()
    cache['global_end_index'].zero_()
    cache['local_end_index'].zero_()
    adapter = CheckedAdapter(SimpleNamespace(blocks=[SimpleNamespace(self_attn=attention)]),router=router,keep_ratio=.5)
    with torch.no_grad(), sdpa_backend(), adapter:
        for chunk in range(3):
            for step in range(3):
                adapter.event = dict(prompt_id='fixture',chunk_id=chunk,step_id=step,timestep=500 if step<2 else 0,
                                     phase='denoise' if step<2 else 'context')
                attention(torch.randn((1,12,128),device=device,dtype=torch.bfloat16),
                          torch.tensor([12],device=device),torch.tensor([[3,2,2]],device=device),
                          rope_params(32,128).to(device),None,kv_cache=cache,current_start=chunk*12)
    assert len(rebuilds)==2, rebuilds
    result['sparse_adapter_matches_statistics'] = True
    result['history_precomputes_for_two_nonempty_versions'] = len(rebuilds)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-events', type=int, default=4)
    p.add_argument('--cache-check', action='store_true')
    p.add_argument('--output')
    args = p.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    model = SparseRouter(RouterConfig(target_size=16)).to(device)
    n,d = 257,128
    sample = dict(q=torch.randn(16,d,device=device), k_hist=torch.randn(n,d,device=device),
                  v_hist=torch.randn(n,d,device=device), k_current=torch.randn(23,d,device=device),
                  v_current=torch.randn(23,d,device=device),
                  token_positions=torch.stack((torch.arange(n,device=device)//64,
                      torch.arange(n,device=device)//8%8,torch.arange(n,device=device)%8),-1))
    result = {'synthetic': check_sample(sample,model)}
    scores = torch.tensor([-.3,.2,.5],dtype=torch.float64,requires_grad=True)
    costs = torch.tensor([2.,3.,5.],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda s,c: soft_budget_relaxation(s,c,4,.3), (scores,costs), eps=1e-5,atol=1e-4)
    assert abs(float((soft_budget_relaxation(scores,costs,4,.3)*costs).sum())-4)<1e-6
    # Skip oversized first candidate rather than stopping the ranking traversal.
    assert select_under_budget(torch.tensor([3.,2.,1.]),torch.tensor([10.,2.,1.]),3).tolist()==[0.,1.,1.]
    result['budget_implicit_gradient_check'] = True
    if args.data:
        dataset = AttentionDataset(args.data)
        if not len(dataset):
            raise ValueError('No captured events')
        result['real'] = [check_sample(to_device(dataset[i],device),model)
                          for i in range(min(len(dataset),args.max_events))]
    if args.cache_check:
        result['cache'] = cache_check(device)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
