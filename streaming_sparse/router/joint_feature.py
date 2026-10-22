import torch
from torch import nn


class JointFeature(nn.Module):
    def __init__(self, head_dim=128, heads=12, layers=30):
        super().__init__()
        self.group = nn.Sequential(nn.Linear(2 * head_dim, 128), nn.GELU(),
                                   nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64))
        self.content = nn.Sequential(nn.Linear(2 * head_dim, 256), nn.GELU(), nn.Linear(256, 128))
        self.head = nn.Embedding(heads, 2 * head_dim)
        self.layer = nn.Embedding(layers, 2 * head_dim)
        self.position = nn.Linear(3, 2 * head_dim, bias=False)

    def forward(self, k, v, positions, head_id, layer_id):
        # Absolute coordinates use a fixed scale, so an unchanged history has unchanged features.
        kv = torch.cat((k.float(), v.float()), -1)
        kv = kv + self.head.weight[head_id] + self.layer.weight[layer_id]
        kv = kv + self.position(positions.float() / 64)
        return self.group(kv), self.content(kv)
