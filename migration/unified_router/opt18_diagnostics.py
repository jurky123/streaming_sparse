"""Read-only Opt18 weight diagnostics; separate from sealed experiment code."""
import argparse
import collections
import gzip
import json
import statistics
import torch
from .common import config, atomic_json
from .opt18.setup import ROOT
from .opt18.measure import Runner, Opt14
from .opt17.data import Events
from .opt17.model import JointGroupedRouter
from .opt17.route import route
from .opt16.grouping import order, gather, restore
from .opt9.segmented_attention import attention
from .opt5.probe import capture, timing


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=['Fast35', 'Quality50'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 256
    _, m = config(ROOT)
    events = Events(m, 24)
    prompt = sorted({x['row']['prompt_id'] for x in events.val})[0]
    item = next(x for x in events.val if x['row']['prompt_id'] == prompt
                and x['row']['history_tokens'] == 14040 and x['row']['step'] == 3)
    model = JointGroupedRouter(m, 24, args.variant).cuda().eval()
    ckpt = ROOT / f'training/layer24/{args.variant}/best.pt'
    model.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=True)['model'])
    runner = Runner(model, events, item, Opt14(m, 24))
    f = model.options['candidate_fraction']
    reps = model.options['representatives']
    kv = runner.builder.current(runner.history, runner.raw['kc'], runner.raw['vc'])
    ix = order(runner.q, model.tree('q'))
    qq = gather(runner.q, ix)
    mask = route(qq, kv.state, model.keep, f, representatives=reps)
    nh = kv.history_tokens
    def attn():
        return attention(qq, kv.k[:, :nh], kv.v[:, :nh], kv.k[:, nh:], kv.v[:, nh:],
                         mask, pack_v=True, fuse_pack=True)
    yy = attn()
    components = {
        'current_KV': lambda: runner.builder.current(runner.history, runner.raw['kc'], runner.raw['vc']),
        'Q_group_gather': lambda: gather(runner.q, order(runner.q, model.tree('q'))),
        'route': lambda: route(qq, kv.state, model.keep, f, representatives=reps),
        'attention': attn,
        'restore': lambda: restore(yy, ix),
        'warm_total': lambda: runner.warm(f, reps),
        'cold_total': lambda: runner.cold(f, reps),
        'dense_total': runner.dense,
    }
    graphs = {k: capture(fn)[0] for k, fn in components.items()}
    for g in graphs.values():
        timing(g, 100)
    samples = {k: [] for k in graphs}
    for repeat in range(4):
        names = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in names:
            samples[name].append(timing(graphs[name], 100))

    def scoped():
        scope = torch.profiler.record_function
        with scope('OPT::current_KV'):
            packed = runner.builder.current(runner.history, runner.raw['kc'], runner.raw['vc'])
        with scope('OPT::Q_group_gather'):
            index = order(runner.q, model.tree('q'))
            query = gather(runner.q, index)
        with scope('OPT::route'):
            selected = route(query, packed.state, model.keep, f, representatives=reps)
        with scope('OPT::attention'):
            n = packed.history_tokens
            y = attention(query, packed.k[:, :n], packed.v[:, :n], packed.k[:, n:], packed.v[:, n:],
                          selected, pack_v=True, fuse_pack=True)
        with scope('OPT::restore'):
            return restore(y, index)

    expected = runner.warm(f, reps)
    for _ in range(3):
        scoped()
    torch.cuda.synchronize()
    folder = ROOT / f'diagnostics/layer24_long/{args.variant}'
    folder.mkdir(parents=True, exist_ok=True)
    repeats = 5
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            actual = scoped()
        torch.cuda.synchronize()
    assert torch.equal(expected, actual)
    trace = folder / 'trace.json'
    prof.export_chrome_trace(str(trace))
    data = json.loads(trace.read_text())['traceEvents']
    with gzip.open(folder / 'trace.json.gz', 'wt') as out:
        json.dump({'traceEvents': data}, out)
    trace.unlink()
    scopes = [e for e in data if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('OPT::')]
    launches = {e['args']['correlation']: e for e in data
                if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'correlation' in e.get('args', {})}
    modules = collections.defaultdict(float)
    kernels = {}
    for e in data:
        if e.get('cat') != 'kernel':
            continue
        api = launches.get(e.get('args', {}).get('correlation'))
        candidates = [s for s in scopes if api and s['tid'] == api['tid']
                      and s['ts'] <= api['ts'] <= s['ts'] + s['dur']]
        name = min(candidates, key=lambda s: s['dur'])['name'][5:] if candidates else 'unattributed'
        ms = e['dur'] / 1000 / repeats
        modules[name] += ms
        key = (name, e['name'])
        row = kernels.setdefault(key, dict(module=name, kernel=e['name'], ms=0,
            registers=e.get('args', {}).get('registers per thread'),
            estimated_occupancy=e.get('args', {}).get('est. achieved occupancy %')))
        row['ms'] += ms
    assert 'unattributed' not in modules, modules
    result = dict(variant=args.variant, layer=24, sample=item['row'], checkpoint=str(ckpt),
                  fraction=f, representatives=reps,
                  graph_ms={k: statistics.median(v) for k, v in samples.items()}, graph_samples=samples,
                  profile_module_kernel_ms=dict(modules),
                  profile_kernels=sorted(kernels.values(), key=lambda x: -x['ms']),
                  exact_profile_output=True,
                  scope='One Val prompt, layer24, step3, history14040/current4680; no full-video inference. Component graph timings are independent, not strictly additive. Profile times are kernel sums, excluding host gaps.')
    atomic_json(folder / 'result.json', result)
    print(json.dumps({k: result[k] for k in ('variant', 'graph_ms', 'profile_module_kernel_ms')}), flush=True)


if __name__ == '__main__':
    main()
