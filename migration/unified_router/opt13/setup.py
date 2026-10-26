import json
from pathlib import Path
from ..common import atomic_json,sha
ROOT=Path('streaming_sparse/v10/experiments/opt13_compact_20261025').resolve()
PARENT=Path('streaming_sparse/v10/experiments/opt11_finetune_20261025').resolve()
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'experiment.json').exists():return
 m=json.loads((PARENT/'experiment.json').read_text());m.update(parent_opt11=str(PARENT),baseline='Opt11 Compact strong-stage best',scope='Compact only; no skipped PV pages, no FP16 PV, no Q64 sharing; independent query routing',variants=['Reference','ParallelSelect','Subgroups','Combined'])
 atomic_json(ROOT/'experiment.json',m)
 locked={str(f.resolve()):sha(f)for dirname in ['opt10','opt11','opt12']for f in (Path('migration/unified_router')/dirname).glob('*.py')}
 for layer in m['layers']:
  f=PARENT/f'strong_training/layer{layer:02d}/Compact/best.pt';locked[str(f)]=sha(f)
 atomic_json(ROOT/'parent_lock.json',locked)
if __name__=='__main__':main()
