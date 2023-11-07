import torch.nn as nn
from Complete.Transformer.MultiHeadSelfAttention import MHSA
from Complete.Transformer.MultiHeadCrossAttention import MHCA
from Complete.Transformer.EasyFeedForward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_att, n_heads, width_FF=[32], dropout_A=0., dropout_FF=0., pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attention = MHSA(d_att, n_heads, dropout=dropout_A)
        self.first_layer_norm = nn.LayerNorm(d_att)
        self.cross_attention = MHCA(d_att, n_heads, dropout=dropout_A)
        self.second_layer_norm = nn.LayerNorm(d_att)
        self.feed_forward = FeedForward(d_att, d_att, widths=width_FF, dropout=dropout_FF)
        self.third_layer_norm = nn.LayerNorm(d_att)

    def forward(self, target, source, mask=None):
        if self.pre_norm:
            y = self.self_attention(self.first_layer_norm(target), mask=mask) + target
            y = self.cross_attention(x_target=self.second_layer_norm(y), x_source=self.second_layer_norm(source)) + y
            y = self.feed_forward(self.third_layer_norm(y)) + y
        else:
            y = self.first_layer_norm(self.self_attention(target, mask=mask) + target)
            y = self.second_layer_norm(self.cross_attention(x_target=y, x_source=source) + y)
            y = self.third_layer_norm(self.feed_forward(y) + y)
        return y

