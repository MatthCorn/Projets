import torch.nn as nn
from PointTransformer.SelfAttention import SA
from Complete.Transformer.EasyFeedForward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_att, d_group, width_FF=[32], dropout_SA=0., dropout_FF=0., norm='post'):
        super().__init__()
        self.norm = norm
        self.self_attention = SA(d_att, d_group=d_group, dropout=dropout_SA)
        self.first_layer_norm = nn.LayerNorm(d_att)
        self.feed_forward = FeedForward(d_att, d_att, widths=width_FF, dropout=dropout_FF)
        self.second_layer_norm = nn.LayerNorm(d_att)

    def forward(self, x, positional_adding_bias, positional_multiplying_bias, mask=None):
        if self.norm == 'pre':
            y = self.self_attention(self.first_layer_norm(x), positional_adding_bias, positional_multiplying_bias, mask) + x
            y = self.feed_forward(self.second_layer_norm(y) + y)
        elif self.norm == 'post':
            y = self.first_layer_norm(self.self_attention(x, positional_adding_bias, positional_multiplying_bias, mask) + x)
            y = self.second_layer_norm(self.feed_forward(y) + y)
        else:
            y = self.self_attention(x, mask) + x
            y = self.feed_forward(y) + y
        return y
