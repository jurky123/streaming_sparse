from pathlib import Path


def protect_output(output):
    output = Path(output).resolve()
    sparse = Path(__file__).resolve().parents[1]
    for name in ('v02', 'router', 'capture', 'attention', 'integration', 'evaluation', 'train'):
        if output.is_relative_to(sparse/name):
            raise ValueError('v0.3 output must not overwrite frozen code/data/results')
    return output
