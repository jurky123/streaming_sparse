"""Retrain all four K-only assignments and the candidate64 projection."""
import argparse
import math
import os
from pathlib import Path
import time
import torch
from torch import nn
from .common import config, save, update, paired_seed, sha


def reconstruction(x, weight, bias):
    h, b, n, d = x.shape
    a = (torch.bmm(x.flatten(1, 2), weight.transpose(1, 2)).reshape(h, b, n, -1)+bias[:, None, None]).softmax(-1)
    count = a.sum(2)
    centers = torch.einsum('hbtp,hbtd->hbpd', a, x)/count.clamp_min(1e-8)[..., None]
    pred = torch.einsum('hbtp,hbpd->hbtd', a, centers)
    loss = (pred-x).square().sum()/x.square().sum().clamp_min(1e-8)
    usage = a.mean((1, 2)).clamp_min(1e-12)
    return loss + .01*(usage*(usage*a.shape[-1]).log()).sum(-1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    root, m = config(args.directory)
    paired_seed(30)
    torch.cuda.set_device(0)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    for layer in m['layers'][rank::world]:
        destination = root/'parameters'/f'layer{layer:02d}.pt'
        if destination.exists():
            continue
        initial = torch.load(root/f'initial_parameters/layer{layer:02d}.pt', map_location='cpu', weights_only=True)
        parameters = {k: v.cuda() for k, v in initial['parameters'].items()}
        trainable = nn.ParameterDict({name: nn.Parameter(value) for name, value in parameters.items()
                                     if name.endswith(('_weight', '_bias'))})
        trainable['projection'] = nn.Parameter(parameters['key_basis'][..., :64].clone())
        optimizer = torch.optim.AdamW(trainable.parameters(), lr=m['foundation_learning_rate'], weight_decay=.01)
        data = torch.load(Path(m['cache'])/'foundation'/f'layer{layer:02d}.pt', map_location='cpu', weights_only=True)
        generator = torch.Generator().manual_seed(300+layer)
        recovery = root/f'foundation/layer{layer:02d}.recovery.pt'
        begin = 0
        if recovery.exists():
            old = torch.load(recovery, map_location='cpu', weights_only=True)
            trainable.load_state_dict(old['parameters'])
            optimizer.load_state_dict(old['optimizer'])
            generator.set_state(old['generator'])
            begin = old['updates']
        started = time.monotonic()
        for step in range(begin+1, m['foundation_updates']+1):
            if (root/'STOP').exists():
                raise RuntimeError('Stop requested')
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            for p in [2, 4, 8, 16]:
                pages = data['pages'][p]
                index = torch.randint(pages.shape[1], (8,), generator=generator)
                x = pages[:, index].cuda().float()
                prefix = 'current' if p == 16 else f'history_{p}'
                losses[str(p)] = reconstruction(x, trainable[prefix+'_weight'], trainable[prefix+'_bias'])
            qi = torch.randint(data['q'].shape[1], (64,), generator=generator)
            qq = data['q'][:, qi].cuda().float()
            keys = data['pages'][16].flatten(1, 2)
            ki = torch.randint(keys.shape[1], (64,), generator=generator)
            kk = keys[:, ki].cuda().float()
            truth = torch.bmm(qq, kk.transpose(1, 2))/128**.5
            u = trainable['projection']
            pred = torch.bmm(torch.bmm(qq, u), torch.bmm(kk, u).transpose(1, 2))/128**.5
            projection_loss = (pred-truth).square().sum()/truth.square().sum().clamp_min(1e-8)
            total = sum(losses.values())/4 + .1*projection_loss
            total.backward()
            if step == begin+1:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                           for p in trainable.values()), 'Every new assignment/projection must receive real gradients'
            grad = torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1., error_if_nonfinite=True)
            lr = m['foundation_learning_rate']*min(1., step/20)*(.05+.95*.5*(1+math.cos(math.pi*step/m['foundation_updates'])))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            with torch.no_grad():
                q, _ = torch.linalg.qr(trainable['projection'])
                trainable['projection'].copy_(q)
            values = dict(updates=step, layer=layer, loss=float(total), reconstruction={k: float(v) for k,v in losses.items()},
                qk_projection_nmse=float(projection_loss), gradient_norm=float(grad), learning_rate=lr,
                elapsed_seconds=time.monotonic()-started)
            if step == begin+1 or step % 10 == 0:
                import json
                log = root/f'foundation/layer{layer:02d}.metrics.jsonl'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a') as f:
                    f.write(json.dumps(values)+'\n')
                update(root/f'foundation/rank{rank}.json', status='training', **values)
            if step % 25 == 0 or step == m['foundation_updates']:
                save(recovery, dict(updates=step, parameters=trainable.state_dict(), optimizer=optimizer.state_dict(), generator=generator.get_state()))
        for name, value in trainable.items():
            if name != 'projection':
                parameters[name] = value.detach()
        parameters['key_basis'][..., :64] = trainable['projection'].detach()
        save(destination, dict(**{k:v for k,v in initial.items() if k!='parameters'},
            parameters={k:v.cpu() for k,v in parameters.items()}, foundation_updates=m['foundation_updates'],
            initialization='fresh, no inherited Router tensors', foundation_sha256=sha(__file__)))
    update(root/f'foundation/rank{rank}.json', status='completed', layers=m['layers'][rank::world])


if __name__ == '__main__':
    main()
