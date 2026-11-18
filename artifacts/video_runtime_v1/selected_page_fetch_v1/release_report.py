import json
from pathlib import Path
from statistics import median

import torch

directory = Path(__file__).parent
dense = torch.load('/tmp/jzk_cuda_samecard_dense/latents.pt', weights_only=True).float()
tiers = {}
for tier in ('base', 'intersection', 'step'):
    dest = Path('/tmp/jzk_route_fetch_final_' + tier)
    run = json.loads((dest / 'result.json').read_text())
    z = torch.load(dest / 'latents.pt', weights_only=True)
    old = torch.load(Path('/tmp/jzk_carry_prepared_final_' + tier) / 'latents.pt', weights_only=True)
    equal = torch.equal(z.view(torch.uint8), old.view(torch.uint8))
    assert equal
    tiers[tier] = dict(seconds=run['seconds'], median_s=run['median_seconds'], bitwise=equal,
                       dense_nmse=float((z.float()-dense).square().sum()/dense.square().sum()))

pairs = {}
for name in ('installed_video', 'early_mask_multi'):
    rows = json.loads((directory / (name+'.json')).read_text())
    gains = [1000*(row['baseline_s']-row['candidate_s']) for row in rows]
    pairs[name] = dict(pairs=len(rows), faster=sum(x>0 for x in gains), median_saved_ms=median(gains),
                       bitwise=all(row['bitwise'] for row in rows))
timeline = json.loads((directory / 'timeline.json').read_text())
times = {name: sum(v for chunk in timeline for k,v in chunk['by_name_ms'].items() if k.startswith(prefix))
         for name,prefix in [('main','exact('),('prepare','prepare('),('guard','retry')]}
reference = sum(json.loads((directory / f'dense_{p}.json').read_text())['reference_ms']*calls
                for p,calls in [(216,116),(288,116),(360,116),(432,464)])
report = dict(tiers=tiers, paired=pairs, device_ms=times, same_card_dense_reference_ms=reference)
(directory/'release.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
