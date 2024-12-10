import torch.nn as nn
from PointTransformer.OpSelfAttention import SA
from PointTransformer.OpCrossAttention import CA
from Complete.Transformer.EasyFeedForward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_att, d_group=1, width_FF=[32], dropout_A=0., dropout_FF=0., norm='post'):
        super().__init__()
        self.norm = norm
        self.self_attention = SA(d_att, d_group=d_group, dropout=dropout_A)
        self.first_layer_norm = nn.LayerNorm(d_att)
        self.cross_attention = CA(d_att, d_group=d_group, dropout=dropout_A)
        self.second_layer_norm = nn.LayerNorm(d_att)
        self.feed_forward = FeedForward(d_att, d_att, widths=width_FF, dropout=dropout_FF)
        self.third_layer_norm = nn.LayerNorm(d_att)

    def forward(self, target, source, positional_adding_bias_tt, positional_multiplying_bias_tt,
                positional_adding_bias_ts, positional_multiplying_bias_ts, mask=None):
        if self.norm == 'pre':
            y = self.self_attention(self.first_layer_norm(target), positional_adding_bias_tt,
                                    positional_multiplying_bias_tt, mask=mask) + target
            y = self.cross_attention(self.second_layer_norm(source), self.second_layer_norm(y),
                                     positional_adding_bias_ts, positional_multiplying_bias_ts) + y
            y = self.feed_forward(self.third_layer_norm(y)) + y
        elif self.norm == 'post':
            y = self.first_layer_norm(self.self_attention(target, positional_adding_bias_tt,
                                                          positional_multiplying_bias_tt, mask=mask) + target)
            y = self.second_layer_norm(self.cross_attention(source, y, positional_adding_bias_ts, positional_multiplying_bias_ts) + y)
            y = self.third_layer_norm(self.feed_forward(y) + y)
        else:
            y = self.self_attention(target, positional_adding_bias_tt,
                                    positional_multiplying_bias_tt, mask=mask) + target
            y = self.cross_attention(source, y, positional_adding_bias_ts, positional_multiplying_bias_ts) + y
            y = self.feed_forward(y) + y
        return y

