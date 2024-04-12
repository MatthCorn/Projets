from Complete.Transformer.EncoderTransformer import EncoderLayer
from Complete.Transformer.EasyFeedForward import FeedForward
import torch.nn as nn
import torch

class Network(nn.Module):
    def __init__(self, d_in=10, d_att=64, n_heads=4):
        super().__init__()
        self.Emb = FeedForward(d_in=d_in, d_out=d_att)
        self.Encoder = EncoderLayer(d_att=3*d_att, n_heads=n_heads, width_FF=[3*d_att])
        self.DecodEmb = FeedForward(d_in=3*d_att, d_out=d_in, widths=[d_att, 16])
        self.DecodRanks = FeedForward(d_in=3*d_att, d_out=1, widths=[d_att, 16])

    def forward(self, x, Ini, Fin):
        x = self.Emb(x)
        x = torch.concatenate([x, Ini, Fin], dim=-1)
        x = self.Encoder(x)
        y = self.DecodEmb(x)
        ranks = self.DecodRanks(x).squeeze(-1)
        return y, ranks

