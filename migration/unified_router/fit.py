"""Fresh Train-only initialization for distance-P2/P4/P8 and current-P16."""
import argparse
import os
from pathlib import Path
import torch
from .common import config, load_raw, save, update, paired_seed, sha


def eig(cov):
    values, vectors = torch.linalg.eigh(cov)
    vectors = vectors.flip(-1).float()
    index = vectors.abs().argmax(-2, keepdim=True)
    sign = vectors.gather(-2, index).sign()
    return values.flip(-1).float(), vectors * sign


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    args = p.parse_args()
    root, m = config(args.directory)
    torch.cuda.set_device(0)
    paired_seed(20)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    train = [r for r in m['rows'] if r['split'] == 'train']
    chosen = []
    for i, pid in enumerate(m['splits']['train']):
        chosen.append(next(r for r in train if r['prompt_id'] == pid and
            r['generation_seed'] == i % 2 and r['history_tokens'] == [4680, 9360, 14040][i % 3] and r['step'] == i % 4))
    for layer in m['layers'][rank::world]:
        dest = root / 'initial_parameters' / f'layer{layer:02d}.pt'
        if dest.exists():
            continue
        cov = torch.zeros(12, 128, 128, device='cuda', dtype=torch.float64)
        group_cov = {p: torch.zeros_like(cov) for p in [2, 4, 8, 16]}
        group_count = {p: 0 for p in group_cov}
        pools = {p: [] for p in group_cov}
        queries = []
        for n, row in enumerate(chosen, 1):
            d = load_raw(root, row, layer)
            for name in ['q', 'kh', 'kc']:
                x = d[name][:, torch.linspace(0, d[name].shape[1]-1, min(256, d[name].shape[1]), device='cuda').long()].double()
                cov += torch.bmm(x.transpose(1, 2), x)/x.shape[1]
            queries.append(d['q'].cpu())
            segments = [(16, d['kc'])]
            for start in range(0, d['kh'].shape[1], 4680):
                age = (d['kh'].shape[1]-start+4679)//4680
                p = 8 if age == 1 else 4 if age == 2 else 2
                segments.append((p, d['kh'][:, start:start+4680]))
            for p, raw in segments:
                x = raw[:, ::max(1, raw.shape[1]//256)].double()
                group_cov[p] += torch.bmm(x.transpose(1, 2), x)/x.shape[1]
                group_count[p] += 1
                pages = raw[:, :raw.shape[1]//64*64].reshape(12, -1, 64, 128)
                indices = torch.linspace(0, pages.shape[1]-1, min(8, pages.shape[1]), device='cuda').long()
                pools[p].append(pages[:, indices].cpu())
            update(root/'fit'/f'rank{rank}.json', status='fitting', layer=layer, completed=n, total=len(chosen))
        _, uk = eig(cov/len(chosen))
        parameters = dict(key_basis=uk.cpu(),
            value_basis=torch.eye(128)[None].expand(12, -1, -1)[..., :32].clone())
        for p in group_cov:
            eigenvalues, directions = eig(group_cov[p]/group_count[p])
            prefix = 'current' if p == 16 else f'history_{p}'
            parameters[prefix+'_weight'] = ((directions[..., :p]/eigenvalues[..., :p].clamp_min(1e-6).sqrt()[:, None]).transpose(1, 2)*.5).cpu()
            parameters[prefix+'_bias'] = torch.zeros(12, p)
        save(Path(m['cache'])/'foundation'/f'layer{layer:02d}.pt',
            dict(pages={p: torch.cat(values, 1) for p, values in pools.items()}, q=torch.cat(queries, 1)))
        save(dest, dict(parameters=parameters, layer=layer, train_sample_ids=[r['sample_id'] for r in chosen],
            key_basis_kind='Train Q/K covariance; candidate64 with128 reference',
            assignment_kind='Fresh Train K second-moment directions, variance-scaled; no inherited Router tensors',
            value_basis_kind='placeholder before action-aware Train fit; not used in P1',
            backbone_source_sha256=m['backbone_source_sha256']))
    update(root/'fit'/f'rank{rank}.json', status='completed', layers=m['layers'][rank::world])


if __name__ == '__main__':
    main()
