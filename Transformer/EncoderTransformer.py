import torch.nn as nn
from Transformer.RelativeMultiHeadAttention import RMHSA
from Transformer.EasyFeedForward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, WidthsFeedForward=[512], max_len=64, MHADropout=0.1, FFDropout=0.1, masked=True):
        super().__init__()
        self.MultiHeadAttention = RMHSA(d_model, num_heads, max_len=max_len, dropout=MHADropout, masked=masked)
        self.FirstLayerNorm = nn.LayerNorm(d_model)
        self.FeedForward = FeedForward(d_model, d_model, widths=WidthsFeedForward, dropout=FFDropout)
        self.SecondLayerNorm = nn.LayerNorm(d_model)

    def forward(self,x):
        y = self.FirstLayerNorm(self.MultiHeadAttention(x) + x)
        y = self.SecondLayerNorm(self.FeedForward(y) + y)
        return y
