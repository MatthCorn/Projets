import torch.nn as nn
from Transformer.RelativeMultiHeadSelfAttention import RMHSA
from Perceiver.RelativeMultiHeadCrossAttentionSoft import RLCA
from Transformer.EasyFeedForward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_att, num_heads, WidthsFeedForward=[512], target_len=64, MHADropout=0.1, FFDropout=0.1, masked=True, relative=True):
        super().__init__()
        self.target_len = target_len
        self.MultiHeadAttention = RMHSA(d_model=d_model, d_att=d_att, num_heads=num_heads, max_len=target_len, dropout=MHADropout, masked=masked, relative=relative)
        self.FirstLayerNorm = nn.LayerNorm(d_model)
        self.CrossAttentionLayer = RLCA(d_latent=d_model, d_input=d_model, d_att=d_att, num_heads=num_heads, RPR_len=target_len,
                                        dropout=MHADropout, masked=False, relative=False)
        self.SecondLayerNorm = nn.LayerNorm(d_model)
        self.FeedForward = FeedForward(d_model, d_model, widths=WidthsFeedForward, dropout=FFDropout)
        self.ThirdLayerNorm = nn.LayerNorm(d_model)

    def forward(self, target, source):
        # target doit toujours être de la forme
        # target.shape = (batch_size, self.max_len, d_att)
        y = self.FirstLayerNorm(self.MultiHeadAttention(target) + target)
        y = self.SecondLayerNorm(self.CrossAttentionLayer(x_latent=y, x_input=source) + y)
        y = self.ThirdLayerNorm(self.FeedForward(y) + y)
        return y

