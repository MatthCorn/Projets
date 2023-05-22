import torch.nn as nn
from Transformer.EncoderTransformer import EncoderLayer as TransformerEncoderLayer
from Perceiver.RelativeMultiHeadCrossAttention import RLCA

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, latent_len=16, WidthsFeedForward=[512], max_len=64, ADropout=0.1, FFDropout=0.1, shared=True, masked=False):
        super().__init__()
        self.FirstCrossAttention = RLCA(d_model, num_heads, latent_len=latent_len, max_len=max_len, dropout=ADropout, masked=False)
        self.FirstTransformer = TransformerEncoderLayer(d_model, num_heads, WidthsFeedForward=WidthsFeedForward,
                                                        max_len=max_len, MHADropout=ADropout, FFDropout=FFDropout, masked=False)
        self.FeedForward = FeedForward(d_model, d_model, widths=WidthsFeedForward, dropout=FFDropout)
        self.SecondLayerNorm = nn.LayerNorm(d_model)

    def forward(self,x):
        y = self.FirstLayerNorm(self.MultiHeadAttention(x) + x)
        y = self.SecondLayerNorm(self.FeedForward(y) + y)
        return y