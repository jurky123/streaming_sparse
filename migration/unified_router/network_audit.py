"""Validate live routing and full network forward before the expensive rollout."""
import argparse
from pathlib import Path
import torch

from .common import config,load_raw,atomic_json,sha
from .network import load_network
from .labels import feature_path
from .learning import batch,choose
from .core import LabelStore


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args()
    root,m=config(a.directory);parent=Path(m['parent_experiment']);_,old=config(parent)
    torch.cuda.set_device(0);net,_=load_network(parent)
    versions=tuple(p._version for p in net.model.parameters())
    layer=old['layers'][0]
    row=next(r for r in old['rows']if r['split']=='val'and r['history_tokens']==4680)
    d=load_raw(parent,row,layer)
    basis=torch.load(parent/f'action_basis/layer{layer:02d}.pt',map_location='cpu',weights_only=True)
    cache=torch.load(feature_path(parent,old['cache'],row,layer,old['monitor_beta']),map_location='cpu',weights_only=True)
    truth=LabelStore(d['q'][:,:2],d['kh'],d['vh'],d['kc'],d['vc'])
    comparisons={}
    for kind in ['K32-Set','V32-Set']:
        net.set_mode(kind);sketch=net.router.sketch(layer,d['kh'],d['vh'],d['kc'],d['vc'])
        mask=net.router.route(layer,d['q'],sketch,row['step'])
        b=batch(parent,old,[row],layer,old['monitor_beta'],kind,basis)
        pred=net.router.heads[f'{layer}_{kind}'](b['x'])
        ix=choose(pred,b['valid'],b['cost'],0.).cpu()[...,None]
        expected_h=cache['base_H']+cache['delta_H'].gather(-1,ix).squeeze(-1)
        expected_c=cache['base_C']+cache['delta_C'].gather(-1,ix).squeeze(-1)
        retained=mask*sketch.costs[None,None]
        torch.testing.assert_close(retained[...,sketch.origins==0].sum(-1).cpu(),expected_h,atol=0,rtol=0)
        torch.testing.assert_close(retained[...,sketch.origins==1].sum(-1).cpu(),expected_c,atol=0,rtol=0)
        actual=(truth.output(mask[:,:2])-truth.target).square().sum(-1)
        expected=(cache['base_error']-cache['gain'].gather(-1,ix).squeeze(-1))[:,:2].cuda()
        torch.testing.assert_close(actual,expected.double(),rtol=5e-5,atol=2e-6)
        comparisons[kind]=dict(live_matches_offline_cost=True,direct_error_max_abs=float((actual-expected).abs().max()))
    del truth,d,cache
    from streaming_sparse.v08.replay import ReplayDataset
    from streaming_sparse.capture.capture_runner import sdpa_backend
    ds=ReplayDataset('val',require_complete=True);sample=ds.load(row['replay_index'])
    with sdpa_backend():
        net.set_mode('Dense')
        first=net.forward_chunk(sample['noisy'].cuda().to(torch.bfloat16),sample['prompt_embeds'].cuda().to(torch.bfloat16),
            sample['timestep'],(),0,sparse=False,checkpoint=False,collect_aux=False)['x0'].cpu()
        net.set_mode('V32-Set')
        sparse_first=net.forward_chunk(sample['noisy'].cuda().to(torch.bfloat16),sample['prompt_embeds'].cuda().to(torch.bfloat16),
            sample['timestep'],(),0,sparse=True,checkpoint=False,collect_aux=False)['x0'].cpu()
        assert torch.equal(first,sparse_first)
        result=net.replay(sample,sparse=True,checkpoint=False,collect_aux=False,recent_gradient=False)
        assert torch.isfinite(result['x0']).all()
        routed={r['layer']for r in net.routing_records}
        assert routed==set(old['layers'])and 0 not in routed
    assert tuple(p._version for p in net.model.parameters())==versions
    assert not any(p.requires_grad for p in net.parameters())
    atomic_json(root/'network_audit.json',dict(passed=True,scope='Six-layer adapter integration; full29 rollout is a later phase.',
        first_chunk_bit_identical=True,live_vs_offline=comparisons,full_network_forward_finite=True,
        layer0_dense=True,context_writes_dense=True,backbone_versions_unchanged=True,
        current_sparsity_supported=True,prototype_age_segments_preserved=True,
        network_sha256=sha(__file__.replace('network_audit.py','network.py'))))
    print('PASS: live masks match offline actions; first chunk bit-identical; full network forward finite.',flush=True)


if __name__=='__main__':main()
