"""Use raw existing Train/Val; keep previously exposed Test out of selection."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from .common import ROOT,HERE,SnapshotData,DEFAULT_DATA,file_sha


def audit():
    # Verify actual cached-attention source instead of assuming generic causal attention.
    source=ROOT/'wan/modules/causal_model.py';text=source.read_text();tree=ast.parse(text)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='attention']
    cached=[n for n in calls if len(n.args)==3 and isinstance(n.args[0],ast.Name) and n.args[0].id=='roped_query']
    if len(cached)!=1 or cached[0].keywords:raise ValueError('Cached attention call changed; inspect real masking/options')
    a=ast.parse((ROOT/'wan/modules/attention.py').read_text())
    fn=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='attention')
    defaults=dict(zip([x.arg for x in fn.args.args][-len(fn.args.defaults):],fn.args.defaults))
    for name,expected in (('softmax_scale',None),('q_scale',None),('causal',False),('dropout_p',0.)):
        if ast.literal_eval(defaults[name])!=expected:raise ValueError('Attention default changed: '+name)
    checkpoint=ROOT/'checkpoints/self_forcing_dmd.pt'
    known=[ROOT/'streaming_sparse/v03/reports',ROOT/'streaming_sparse/v04/reports']
    return dict(post_rope_qk=True,logit_scale=128**-.5,additional_logit_bias=False,
        visibility='All passed historical and current chunk keys, block-wise causality; no within-chunk token-causal mask',
        explicit_visible_mask='All-true reconstructed from audited cached call; no hidden future cache rows are passed',
        actual_call_line=cached[0].lineno,source_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in
         (source,ROOT/'wan/modules/attention.py',ROOT/'streaming_sparse/integration/self_forcing_adapter.py',ROOT/'streaming_sparse/v02/capture.py')},
        backbone=dict(logical=str(checkpoint),exists=checkpoint.exists(),size=checkpoint.stat().st_size if checkpoint.exists() else None,
                      expected_sha256='a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3'))


def data(split):
    if split=='test':raise ValueError('Old Test is exposed. Independent final Test requires separately locked fresh prompts; Train/Val continue.')
    return SnapshotData(DEFAULT_DATA,split)


def validate_event(event):
    required=('q','k_hist','v_hist','k_current','v_current','o_dense','query_indices','query_positions','current_positions','token_positions')
    for key in required:
        if key not in event:raise ValueError('Missing raw tensor '+key)
        if not torch.isfinite(event[key]).all():raise FloatingPointError('Nonfinite raw data')
    if len(event['query_indices'].unique())!=len(event['query_indices']):raise ValueError('Duplicate query token IDs')
    if not torch.equal(event['current_positions'][event['query_indices']],event['query_positions']):raise ValueError('Q/current position mismatch')
    if not len(event['k_current']):raise ValueError('No visible current KV')
    # Explicit masks: captured cached attention sees the complete current block.
    return dict(all_historical_visible=True,all_current_visible=True,current_tokens=len(event['k_current']),
                noncontiguous_query_indices=bool((event['query_indices'][1:]!=event['query_indices'][:-1]+1).any()))
