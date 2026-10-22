import torch
from torch import nn


class BlockAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 256)
        self.metadata = nn.Linear(4, 256)

    def forward(self, content, assignment, hard, positions):
        costs = assignment.sum(0)
        mean = assignment.T @ content / costs[:, None].clamp_min(1e-6)
        labels = hard.argmax(-1)
        maximum = content.new_full((hard.shape[1], content.shape[1]), -torch.inf)
        maximum = maximum.scatter_reduce(0, labels[:, None].expand_as(content), content,
                                         reduce='amax', include_self=True)
        empty = hard.sum(0) == 0
        maximum = maximum.masked_fill(empty[:, None], 0)
        pos = assignment.T @ (positions.float() / 64) / costs[:, None].clamp_min(1e-6)
        meta = torch.cat((costs.clamp_min(0).log1p()[:, None], pos), -1)
        return self.projection(torch.cat((mean, maximum), -1)) + self.metadata(meta), empty
