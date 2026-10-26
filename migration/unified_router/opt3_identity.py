"""Check aggregate/Gram identities on real prototypes; not a timing kernel."""
import argparse
import math
from pathlib import Path
import torch
from .common import atomic_json
from .opt3_tiled import score_only
from .fast_route import FastRoute


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    root = Path(parser.parse_args().directory)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = torch.load(root/'optimization/stage3_fused_K32/actual_block.pt', map_location='cuda', weights_only=False)
    q = data['q'][:, :72]
    score, raw = score_only(q, *data['prepared'])
    mask = FastRoute(True).mask(score, data['beta'])
    explicit, energy, all_mean, selected_mean = [], [], 0., 0.
    cost = data['sketch'].costs.double()
    offset = 0
    for logits, (_, lc, value) in zip(raw, data['prepared'][1]):
        h, pages, protos = lc.shape
        w = torch.softmax(logits.double().reshape(h,72,pages,protos)/math.sqrt(128) + lc[:,None].double(), -1)
        v = value.double()
        content = torch.einsum('hqbp,hbpd->hqbd', w, v)
        gram = v @ v.transpose(-1,-2)
        norm = torch.einsum('hqbp,hbpr,hqbr->hqb', w, gram, w)
        explicit.append(content); energy.append(norm)
        weighted = w * cost[None,None,offset:offset+pages,None]
        selected = weighted * mask[:,:,offset:offset+pages,None]
        all_mean = all_mean + torch.bmm(weighted.flatten(2),v.flatten(1,2))
        selected_mean = selected_mean + torch.bmm(selected.flatten(2),v.flatten(1,2))
        offset += pages
    content = torch.cat(explicit, 2)
    norm = torch.cat(energy, 2)
    result = dict(scope='Real L24 H14040 K32 first72 queries x12 heads; FP64 algebra check, not performance or video validation',
                  prototype_gram_max_error=float((norm-content.square().sum(-1)).abs().max()), sets={})
    for name, weights, numerator in [('all', cost.expand_as(score), all_mean), ('selected', cost*mask, selected_mean)]:
        den = weights.sum(-1)
        mean = numerator/den[...,None]
        direct_mean = (content*weights[...,None]).sum(2)/den[...,None]
        variance = (norm*weights).sum(-1)/den-mean.square().sum(-1)
        direct_variance = ((content-direct_mean[:,:,None]).square().sum(-1)*weights).sum(-1)/den
        error = dict(mean_max=float((mean-direct_mean).abs().max()),variance_max=float((variance-direct_variance).abs().max()))
        assert error['mean_max'] < 1e-10 and error['variance_max'] < 1e-9
        result['sets'][name] = error
    atomic_json(root/'optimization3/algebra_check.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
