from dataclasses import dataclass
import torch


@dataclass
class BlockStatistics:
    z: torch.Tensor
    u: torch.Tensor
    z_current: torch.Tensor
    u_current: torch.Tensor

    def output(self, mask):
        # A single mask is shared by all queries in this tile.
        denominator = self.z_current + self.z @ mask.float()
        numerator = self.u_current + torch.einsum('qbd,b->qd', self.u, mask.float())
        return numerator / denominator[:, None].clamp_min(1e-30)

    def dense(self):
        return self.output(torch.ones(self.z.shape[-1], device=self.z.device))


def differentiable_block_stats(q, k_hist, v_hist, assignment, k_current, v_current,
                               key_chunk=1024):
    """FP32 two-pass stable statistics, no full Q x N x D or N x B x D tensor.

    `assignment` is deliberately attached to autograd. Q/K/V are frozen data.
    Both history and current keys contribute to the SAME per-query stabilizer.
    """
    q, kh, vh, kc, vc = [x.float() for x in (q, k_hist, v_hist, k_current, v_current)]
    if len(kc) == 0:
        raise ValueError('Always-retained current KV must be nonempty')
    scale = q.shape[-1] ** -0.5
    maxima = q.new_full((len(q),), -torch.inf)
    with torch.no_grad():
        for keys in (kh, kc):
            for start in range(0, len(keys), key_chunk):
                logits = (q @ keys[start:start + key_chunk].T) * scale
                maxima = torch.maximum(maxima, logits.amax(-1))
    b = assignment.shape[-1]
    z = q.new_zeros((len(q), b))
    u = q.new_zeros((len(q), b, vh.shape[-1]))
    z0 = q.new_zeros(len(q))
    u0 = q.new_zeros((len(q), vc.shape[-1]))
    for start in range(0, len(kh), key_chunk):
        end = start + key_chunk
        a = ((q @ kh[start:end].T) * scale - maxima[:, None]).exp()
        memberships = assignment[start:end].float()
        z = z + a @ memberships
        # This intermediate is bounded by Q_tile * key_chunk * head_dim.
        # Batched GEMM avoids a Python loop and preserves the assignment gradient.
        weighted_values = a[:, :, None] * vh[start:end][None]
        u = u + torch.matmul(memberships.T[None], weighted_values)
    for start in range(0, len(kc), key_chunk):
        a = ((q @ kc[start:start + key_chunk].T) * scale - maxima[:, None]).exp()
        z0 = z0 + a.sum(-1)
        u0 = u0 + a @ vc[start:start + key_chunk]
    return BlockStatistics(z, u, z0, u0)


@torch.no_grad()
def exact_attention(q, k, v, query_chunk=16):
    outputs = []
    for tile in q.split(query_chunk):
        logits = tile.float() @ k.float().T / q.shape[-1] ** 0.5
        outputs.append(logits.softmax(-1) @ v.float())
    return torch.cat(outputs)
