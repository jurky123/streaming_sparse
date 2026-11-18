"""Warp instruction slopes for a single extra selected page or full compensation tile."""
import csv
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
with (root / 'counters.csv').open() as f:
    rows = [r for r in csv.DictReader(line for line in f if not line.startswith('==')) if r['ID']]
assert len(rows) == 11
grid = dict(zip(((p, b) for p in (32, 48, 64) for b in (6, 16, 26)), rows[2:]))
result = {}
for metric in rows[0]:
    if not metric.startswith(('gpu__time_duration.', 'smsp__inst_executed')):
        continue
    def value(p, b):
        return float(grid[p, b][metric])
    time_metric = metric.startswith('gpu')
    divisor = 1000 if time_metric else 864 * 4
    page = mean((value(64,b)-value(32,b))/32 for b in (6,16,26))/divisor
    tile = mean((value(p,26)-value(p,6))/20 for p in (32,48,64))/divisor
    result[metric] = dict(page=page, tile=tile)
(root / 'slopes.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
