from torch import nn


class QueryDecoder(nn.Module):
    def __init__(self, head_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(head_dim, 256), nn.GELU(), nn.Linear(256, 256))

    def forward(self, q, h):
        return h @ self.mlp(q.float().mean(0)) / 16
