"""Require the preceding real-data gates before escalating experiment scale."""
import json
from pathlib import Path
from .evaluate import source_identity


def require_training_gates(config, fusion):
    source = source_identity()
    reports = Path(config.get('gate_reports', 'streaming_sparse/v03/reports'))
    correctness = json.loads((reports/'real_correctness.json').read_text())
    if (not correctness.get('real_qkv') or correctness.get('source_sha256') != source or
        not correctness['fusions'][fusion]['passed']):
        raise ValueError('Require a current-source real-QKV numeric/gradient gate before pilot/Full')
    path = Path(config.get('overfit_runs', 'streaming_sparse/v03/runs/overfit'))/fusion/'overfit_gate.json'
    overfit = json.loads(path.read_text())
    if (not overfit.get('real_qkv') or not overfit.get('passed') or
        overfit.get('fusion') != fusion or overfit.get('source_sha256') != source):
        raise ValueError('Single-snapshot hard-error overfit gate must pass before pilot/Full')
    if config.get('stage') == 'full':
        gate = json.loads((reports/'architecture_gate.json').read_text())
        if not gate.get('capture_allowed') or gate.get('source_sha256') != source:
            raise ValueError('4-head architecture gate must pass before Full 12-head training')
