from torch import nn


class BlockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(256)

    def forward(self, blocks, empty):
        return self.norm(self.encoder(blocks[None], src_key_padding_mask=empty[None])[0])
