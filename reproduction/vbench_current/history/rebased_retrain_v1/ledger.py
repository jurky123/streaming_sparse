import json
from pathlib import Path
from collections import Counter

root=Path(__file__).parent
prior=root.parent/'rebased_budget_ablation_v1'
roof=json.loads((prior/'roof.json').read_text())
rows=[]
for f in sorted(root.glob('records/*/*/status.json')):
    v=json.loads(f.read_text()); n=v['records']
    # FP64 exact QK and AV plus four representative QK logits for mass centering.
    work=348*(216+432*3)*2*64*128*(64*2+4)
    ref=work/(roof['torch.float64']['tflops']*1e12)
    measured=sum(v['capture_seconds'])
    rows.append(dict(path=str(f.parent.relative_to(root)),module='teacher_capture',events=n,flop=work,reference_s=ref,measured_s=measured,wall_s=v['wall_seconds'],gap_s=measured-ref,scope='FP64 matrix work only; measured includes copies, serialization, Dense reconstruction validation'))
model=json.loads((prior/'partial_module_ledger.json').read_text())
for f in root.glob('[01]_*_*.json'):
    v=json.loads(f.read_text()); name=f.stem
    idx=name.split('_')[1]
    if name.endswith('_dense') or '_repeat' in name:continue
    for base in model:
        if base['variant']!='closed_'+idx:continue
        stage=base['module']
        ms=v['stages'][stage]['milliseconds'];ref=base['partial_reference_ms']
        rows.append(dict(path=name,module=stage,flop=base['known_flops'],reference_ms=ref,measured_ms=ms,delta_ms=ms-ref if base['measured_minus_partial_ms'] is not None else None,scope=base['scope']))
    rows.append(dict(path=name,module='attention',measured_ms=v['stages']['attention']['milliseconds'],scope='Includes all attention scopes; sparse fusion by window has a separate shape reference in prior module_rows.json'))
modules=json.loads((prior/'module_rows.json').read_text())
for w in modules:
    if w['spec'] not in ([10,0.25],[15,0.25]): continue
    # Q64, Page64, dual AV and four representative keys per page.
    calls=29*4*(36 if w['pages']==432 else 1)
    rows.append(dict(module='fused_attention_same_shape',source='prior isolated module; unchanged kernel and quota dimensions',pages=w['pages'],budget=w['spec'][0],calls=calls,
                     flop=w['exact_flops']+w['comp_flops'],measured_ms=w['fused_ms'],reference_ms=w['at_dense_rate_ms'],reference_tflops=w['dense_flops']/w['dense_ms']/1e9,delta_ms=w['gap_ms']))
(root/'ledger.json').write_text(json.dumps(rows,indent=2))
