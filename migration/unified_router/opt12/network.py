"""Deploy joint checkpoints with unchanged optimized Opt10/Opt11 kernels."""
import os
from pathlib import Path
import torch
from ..common import config
from ..opt10.network import load as parent_load
from ..opt11.network import configure as configure_parent


def configure(net, root, variant, trained=True, base_prepare=None):
    root, m = config(root)
    configure_parent(net, m['parent_opt11'], variant, True, base_prepare)
    if not hasattr(net, 'opt12_initial_parameters'):
        net.opt12_initial_parameters = {
            str(layer): {k: v.detach().clone() for k, v in net.router.states[str(layer)].dictionary().items()}
            for layer in net.layers}
    for layer in net.layers:
        if trained:
            f = root / f'training/layer{layer:02d}/{variant}/best.pt'
            state = torch.load(f, map_location='cpu', weights_only=True)
            assert state['best_metric'] == 'actual_deployment_attention_nmse'
            net.router.heads[f'{layer}_K32-Set'].load_state_dict(state['head'])
            params = state['parameters']
        else:
            params = net.opt12_initial_parameters[str(layer)]
        with torch.no_grad():
            for key, value in net.router.states[str(layer)].dictionary().items():
                value.copy_(params[key])
    net._eval_sketch.clear()
    net.route_cache = {}
    net.opt12_variant, net.opt12_trained = variant, trained
    return net


def load(directory, training_seed=0, **options):
    root, m = config(directory)
    net, _ = parent_load(m['parent_experiment'], training_seed, **options)
    configure(net, root, os.environ.get('SLIMAV_OPT12', 'Aggressive'),
              os.environ.get('SLIMAV_OPT12_TRAINED', '1') == '1')
    return net, m


def main():
    from .. import fast_bench
    fast_bench.load_fast = load
    fast_bench.main()


if __name__ == '__main__':
    main()
