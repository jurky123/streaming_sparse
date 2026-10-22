"""Plan supplemental capture of the eight missing heads, using the v0.2 sampler."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .evaluate import source_identity
from .paths import protect_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--gate', default='streaming_sparse/v03/reports/architecture_gate.json')
    p.add_argument('--execute', action='store_true', help='Explicitly start capture after the real-data architecture gate')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).parent
    protocol = json.loads((here/'configs/protocol.json').read_text())
    prompts = here/'configs/capture_prompts.txt'
    target = protect_output(args.output)
    missing_heads = [i for i in range(12) if i not in (0, 3, 6, 9)]
    argv = [sys.executable, '-m', 'streaming_sparse.v02.capture', '--mode', 'capture',
            '--prompts', str(prompts), '--limit', '24', '--frames', '12', '--seeds', '0', '1',
            '--layers', '4', '14', '24', '--heads', *map(str, missing_heads),
            '--query-tile', '16', '--tiles-per-event', '8', '--backend', 'sdpa', '--output', str(target)]
    print(shlex.join(argv), flush=True)
    if not args.execute:
        print('Plan only. No QKV captured; existing heads 0/3/6/9 will not be overwritten.')
        return
    gate = json.loads(Path(args.gate).read_text())
    if not gate.get('capture_allowed') or gate.get('source_sha256') != source_identity():
        raise ValueError('Current-source real-data architecture gate must pass before supplemental capture')
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Supplemental capture requires a fresh output directory')
    text = '\n'.join(p['text'] for p in protocol['prompts'])+'\n'
    if prompts.exists() and prompts.read_text() != text:
        raise ValueError('Frozen capture prompt text changed')
    prompts.write_text(text)
    subprocess.run(argv, cwd=root, check=True)


if __name__ == '__main__':
    main()
