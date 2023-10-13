import torch.nn as nn
from Complete.Transformer.MultiHeadSelfAttention import MHSA
from Complete.Transformer.EasyFeedForward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_att, n_heads, width_FF=[32], dropout_SA=0., dropout_FF=0.):
        super().__init__()
        self.self_attention = MHSA(d_att, n_heads, dropout=dropout_SA)
        self.first_layer_norm = nn.LayerNorm(d_att)
        self.feed_forward = FeedForward(d_att, d_att, widths=width_FF, dropout=dropout_FF)
        self.second_layer_norm = nn.LayerNorm(d_att)

    def forward(self, x, mask=None):
        y = self.first_layer_norm(self.self_attention(x, mask) + x)
        y = self.second_layer_norm(self.feed_forward(y) + y)
        return y
