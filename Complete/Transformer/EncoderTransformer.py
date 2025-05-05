import torch.nn as nn
from Complete.Transformer.MultiHeadSelfAttention import MHSA
from Complete.Transformer.EasyFeedForward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_att, n_heads, width_FF=[32], dropout_SA=0., dropout_FF=0., norm='post'):
        super().__init__()
        self.norm = norm
        if self.norm != 'none':
            self.first_layer_norm = nn.LayerNorm(d_att)
            self.second_layer_norm = nn.LayerNorm(d_att)
        self.self_attention = MHSA(d_att, n_heads, dropout=dropout_SA)
        self.feed_forward = FeedForward(d_att, d_att, widths=width_FF, dropout=dropout_FF)

    def forward(self, x, mask=None, RoPE=lambda u: u):
        if self.norm == 'pre':
            y = self.self_attention(self.first_layer_norm(x), mask, RoPE) + x
            y = self.feed_forward(self.second_layer_norm(y) + y)
        elif self.norm == 'post':
            y = self.first_layer_norm(self.self_attention(x, mask, RoPE) + x)
            y = self.second_layer_norm(self.feed_forward(y) + y)
        else:
            y = self.self_attention(x, mask, RoPE) + x
            y = self.feed_forward(y) + y
        return y
