"""Explicit replaceable weight provenance; broadcast placeholders are quality-unvalidated."""
from pathlib import Path
import torch
from ..review_v1.model import Model
from ..e29_local_reader.core import LocalReader
from ..e27_adaptive_query.core import AdaptivePhi
from ..learned_cache.io import digest
ROOT=Path(__file__).resolve().parents[2]

@torch.no_grad()
def load(seed=17,checkpoint=None,source_layer=15,source_head=0,reader_checkpoint=None,include_reader=True):
    base=ROOT/'artifacts/four_gaps_20261031/shared/epoch1.pt'
    pre=ROOT/'artifacts/review_v1_20261102/summary_expand/reef_fish/latest.pt'
    model=Model(str(base));state=torch.load(pre,map_location='cuda',weights_only=False)['model']
    if checkpoint:
        path=Path(checkpoint);rec=torch.load(path,map_location='cuda',weights_only=False)
        if 'model' not in rec:raise ValueError('Expected native deployment checkpoint with model field')
        model.load_state_dict(rec['model']);placeholder=False
    else:
        path=ROOT/f'artifacts/e24_joint_representatives_v1/training/C_s{seed}/selected.pt'
        saved=torch.load(path,map_location='cuda',weights_only=False)['layers'];prefix=f'l{source_layer}h{source_head}'
        for l in range(1,30):
            for name in ('pool_k','pool_v','pool_feature'):
                target=state[f'layers.{l}.{name}'];state[f'layers.{l}.{name}']=saved[f'{prefix}.{name}'].expand_as(target).clone()
        model.load_state_dict(state);placeholder=True
    if not include_reader:
        model.eval().requires_grad_(False)
        identity=dict(seed=seed,summary_sha=digest(path),base_sha=digest(pre),placeholder=placeholder,
                      broadcast_source=[source_layer,source_head] if placeholder else None,reader_loaded=False)
        return model,None,None,None,identity
    reader_path=Path(reader_checkpoint) if reader_checkpoint else ROOT/f'artifacts/e48_tail_state_v1/s{seed}/fullset_final.pt'
    reader=LocalReader().cuda();rec=torch.load(reader_path,map_location='cuda',weights_only=False);reader.load_state_dict(rec.get('model',rec))
    phi_path=ROOT/f'artifacts/e27_adaptive_query_v1/s{seed}/selected.pt'
    phi=AdaptivePhi().cuda();phi.load_state_dict(torch.load(phi_path,map_location='cuda',weights_only=False)['model'])
    fit=ROOT/'artifacts/e24_joint_representatives_v1/fit.pt';B=torch.load(fit,weights_only=True)['B'].cuda()
    for net in (model,reader,phi):net.eval().requires_grad_(False)
    identity=dict(seed=seed,summary_sha=digest(path),reader_sha=digest(reader_path),phi_sha=digest(phi_path),B16_sha=digest(fit),base_sha=digest(pre),
                  placeholder=placeholder,broadcast_source=[source_layer,source_head] if placeholder else None,
                  quality_status='engineering substitute; not full-coverage quality validated' if placeholder else 'Reader adaptation and video quality require validation')
    return model,reader,phi,B,identity
